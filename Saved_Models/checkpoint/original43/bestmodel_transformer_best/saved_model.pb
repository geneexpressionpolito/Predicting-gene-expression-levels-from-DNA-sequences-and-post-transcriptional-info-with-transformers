≈Ћ)
°ч
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
Љ
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
≠
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
Н
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
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Н
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
dtypetypeИ
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
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
list(type)(0И
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

2	Р
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Њ
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
executor_typestring И
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
ц
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.1-0-g85c8b2a817f8ж¬$
П
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_4/gamma
И
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_4/beta
Ж
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_4/moving_mean
Ф
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:А*
dtype0
£
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_4/moving_variance
Ь
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:А*
dtype0
{
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И@* 
shared_namedense_26/kernel
t
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes
:	И@*
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
:@@* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:@@*
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:@*
dtype0
z
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

:@*
dtype0
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_28/bias
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
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
«
5token_and_position_embedding_4/embedding_8/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*F
shared_name75token_and_position_embedding_4/embedding_8/embeddings
ј
Itoken_and_position_embedding_4/embedding_8/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_4/embedding_8/embeddings*
_output_shapes
:	А*
dtype0
»
5token_and_position_embedding_4/embedding_9/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДRА*F
shared_name75token_and_position_embedding_4/embedding_9/embeddings
Ѕ
Itoken_and_position_embedding_4/embedding_9/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_4/embedding_9/embeddings* 
_output_shapes
:
ДRА*
dtype0
–
7transformer_block_9/multi_head_attention_9/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*H
shared_name97transformer_block_9/multi_head_attention_9/query/kernel
…
Ktransformer_block_9/multi_head_attention_9/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_9/multi_head_attention_9/query/kernel*$
_output_shapes
:АА*
dtype0
«
5transformer_block_9/multi_head_attention_9/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*F
shared_name75transformer_block_9/multi_head_attention_9/query/bias
ј
Itransformer_block_9/multi_head_attention_9/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_9/multi_head_attention_9/query/bias*
_output_shapes
:	А*
dtype0
ћ
5transformer_block_9/multi_head_attention_9/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*F
shared_name75transformer_block_9/multi_head_attention_9/key/kernel
≈
Itransformer_block_9/multi_head_attention_9/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_9/multi_head_attention_9/key/kernel*$
_output_shapes
:АА*
dtype0
√
3transformer_block_9/multi_head_attention_9/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*D
shared_name53transformer_block_9/multi_head_attention_9/key/bias
Љ
Gtransformer_block_9/multi_head_attention_9/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_9/multi_head_attention_9/key/bias*
_output_shapes
:	А*
dtype0
–
7transformer_block_9/multi_head_attention_9/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*H
shared_name97transformer_block_9/multi_head_attention_9/value/kernel
…
Ktransformer_block_9/multi_head_attention_9/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_9/multi_head_attention_9/value/kernel*$
_output_shapes
:АА*
dtype0
«
5transformer_block_9/multi_head_attention_9/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*F
shared_name75transformer_block_9/multi_head_attention_9/value/bias
ј
Itransformer_block_9/multi_head_attention_9/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_9/multi_head_attention_9/value/bias*
_output_shapes
:	А*
dtype0
ж
Btransformer_block_9/multi_head_attention_9/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*S
shared_nameDBtransformer_block_9/multi_head_attention_9/attention_output/kernel
я
Vtransformer_block_9/multi_head_attention_9/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_9/multi_head_attention_9/attention_output/kernel*$
_output_shapes
:АА*
dtype0
ў
@transformer_block_9/multi_head_attention_9/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*Q
shared_nameB@transformer_block_9/multi_head_attention_9/attention_output/bias
“
Ttransformer_block_9/multi_head_attention_9/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_9/multi_head_attention_9/attention_output/bias*
_output_shapes	
:А*
dtype0
|
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_24/kernel
u
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_24/bias
l
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes	
:А*
dtype0
|
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_25/kernel
u
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_25/bias
l
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes	
:А*
dtype0
є
0transformer_block_9/layer_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*A
shared_name20transformer_block_9/layer_normalization_18/gamma
≤
Dtransformer_block_9/layer_normalization_18/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_9/layer_normalization_18/gamma*
_output_shapes	
:А*
dtype0
Ј
/transformer_block_9/layer_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*@
shared_name1/transformer_block_9/layer_normalization_18/beta
∞
Ctransformer_block_9/layer_normalization_18/beta/Read/ReadVariableOpReadVariableOp/transformer_block_9/layer_normalization_18/beta*
_output_shapes	
:А*
dtype0
є
0transformer_block_9/layer_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*A
shared_name20transformer_block_9/layer_normalization_19/gamma
≤
Dtransformer_block_9/layer_normalization_19/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_9/layer_normalization_19/gamma*
_output_shapes	
:А*
dtype0
Ј
/transformer_block_9/layer_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*@
shared_name1/transformer_block_9/layer_normalization_19/beta
∞
Ctransformer_block_9/layer_normalization_19/beta/Read/ReadVariableOpReadVariableOp/transformer_block_9/layer_normalization_19/beta*
_output_shapes	
:А*
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
Э
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_4/gamma/m
Ц
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_4/beta/m
Ф
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И@*'
shared_nameAdam/dense_26/kernel/m
В
*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes
:	И@*
dtype0
А
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
И
Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_27/kernel/m
Б
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*
_output_shapes

:@@*
dtype0
А
Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_27/bias/m
y
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes
:@*
dtype0
И
Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_28/kernel/m
Б
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m*
_output_shapes

:@*
dtype0
А
Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_28/bias/m
y
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes
:*
dtype0
’
<Adam/token_and_position_embedding_4/embedding_8/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*M
shared_name><Adam/token_and_position_embedding_4/embedding_8/embeddings/m
ќ
PAdam/token_and_position_embedding_4/embedding_8/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_4/embedding_8/embeddings/m*
_output_shapes
:	А*
dtype0
÷
<Adam/token_and_position_embedding_4/embedding_9/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДRА*M
shared_name><Adam/token_and_position_embedding_4/embedding_9/embeddings/m
ѕ
PAdam/token_and_position_embedding_4/embedding_9/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_4/embedding_9/embeddings/m* 
_output_shapes
:
ДRА*
dtype0
ё
>Adam/transformer_block_9/multi_head_attention_9/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*O
shared_name@>Adam/transformer_block_9/multi_head_attention_9/query/kernel/m
„
RAdam/transformer_block_9/multi_head_attention_9/query/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_9/multi_head_attention_9/query/kernel/m*$
_output_shapes
:АА*
dtype0
’
<Adam/transformer_block_9/multi_head_attention_9/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*M
shared_name><Adam/transformer_block_9/multi_head_attention_9/query/bias/m
ќ
PAdam/transformer_block_9/multi_head_attention_9/query/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_9/multi_head_attention_9/query/bias/m*
_output_shapes
:	А*
dtype0
Џ
<Adam/transformer_block_9/multi_head_attention_9/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*M
shared_name><Adam/transformer_block_9/multi_head_attention_9/key/kernel/m
”
PAdam/transformer_block_9/multi_head_attention_9/key/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_9/multi_head_attention_9/key/kernel/m*$
_output_shapes
:АА*
dtype0
—
:Adam/transformer_block_9/multi_head_attention_9/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*K
shared_name<:Adam/transformer_block_9/multi_head_attention_9/key/bias/m
 
NAdam/transformer_block_9/multi_head_attention_9/key/bias/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_9/multi_head_attention_9/key/bias/m*
_output_shapes
:	А*
dtype0
ё
>Adam/transformer_block_9/multi_head_attention_9/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*O
shared_name@>Adam/transformer_block_9/multi_head_attention_9/value/kernel/m
„
RAdam/transformer_block_9/multi_head_attention_9/value/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_9/multi_head_attention_9/value/kernel/m*$
_output_shapes
:АА*
dtype0
’
<Adam/transformer_block_9/multi_head_attention_9/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*M
shared_name><Adam/transformer_block_9/multi_head_attention_9/value/bias/m
ќ
PAdam/transformer_block_9/multi_head_attention_9/value/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_9/multi_head_attention_9/value/bias/m*
_output_shapes
:	А*
dtype0
ф
IAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*Z
shared_nameKIAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/m
н
]Adam/transformer_block_9/multi_head_attention_9/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/m*$
_output_shapes
:АА*
dtype0
з
GAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*X
shared_nameIGAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/m
а
[Adam/transformer_block_9/multi_head_attention_9/attention_output/bias/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_24/kernel/m
Г
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_24/bias/m
z
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_25/kernel/m
Г
*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_25/bias/m
z
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes	
:А*
dtype0
«
7Adam/transformer_block_9/layer_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*H
shared_name97Adam/transformer_block_9/layer_normalization_18/gamma/m
ј
KAdam/transformer_block_9/layer_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_9/layer_normalization_18/gamma/m*
_output_shapes	
:А*
dtype0
≈
6Adam/transformer_block_9/layer_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*G
shared_name86Adam/transformer_block_9/layer_normalization_18/beta/m
Њ
JAdam/transformer_block_9/layer_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_9/layer_normalization_18/beta/m*
_output_shapes	
:А*
dtype0
«
7Adam/transformer_block_9/layer_normalization_19/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*H
shared_name97Adam/transformer_block_9/layer_normalization_19/gamma/m
ј
KAdam/transformer_block_9/layer_normalization_19/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_9/layer_normalization_19/gamma/m*
_output_shapes	
:А*
dtype0
≈
6Adam/transformer_block_9/layer_normalization_19/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*G
shared_name86Adam/transformer_block_9/layer_normalization_19/beta/m
Њ
JAdam/transformer_block_9/layer_normalization_19/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_9/layer_normalization_19/beta/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_4/gamma/v
Ц
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_4/beta/v
Ф
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes	
:А*
dtype0
Й
Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И@*'
shared_nameAdam/dense_26/kernel/v
В
*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes
:	И@*
dtype0
А
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
И
Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_27/kernel/v
Б
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*
_output_shapes

:@@*
dtype0
А
Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_27/bias/v
y
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes
:@*
dtype0
И
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_28/kernel/v
Б
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v*
_output_shapes

:@*
dtype0
А
Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_28/bias/v
y
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes
:*
dtype0
’
<Adam/token_and_position_embedding_4/embedding_8/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*M
shared_name><Adam/token_and_position_embedding_4/embedding_8/embeddings/v
ќ
PAdam/token_and_position_embedding_4/embedding_8/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_4/embedding_8/embeddings/v*
_output_shapes
:	А*
dtype0
÷
<Adam/token_and_position_embedding_4/embedding_9/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДRА*M
shared_name><Adam/token_and_position_embedding_4/embedding_9/embeddings/v
ѕ
PAdam/token_and_position_embedding_4/embedding_9/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_4/embedding_9/embeddings/v* 
_output_shapes
:
ДRА*
dtype0
ё
>Adam/transformer_block_9/multi_head_attention_9/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*O
shared_name@>Adam/transformer_block_9/multi_head_attention_9/query/kernel/v
„
RAdam/transformer_block_9/multi_head_attention_9/query/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_9/multi_head_attention_9/query/kernel/v*$
_output_shapes
:АА*
dtype0
’
<Adam/transformer_block_9/multi_head_attention_9/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*M
shared_name><Adam/transformer_block_9/multi_head_attention_9/query/bias/v
ќ
PAdam/transformer_block_9/multi_head_attention_9/query/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_9/multi_head_attention_9/query/bias/v*
_output_shapes
:	А*
dtype0
Џ
<Adam/transformer_block_9/multi_head_attention_9/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*M
shared_name><Adam/transformer_block_9/multi_head_attention_9/key/kernel/v
”
PAdam/transformer_block_9/multi_head_attention_9/key/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_9/multi_head_attention_9/key/kernel/v*$
_output_shapes
:АА*
dtype0
—
:Adam/transformer_block_9/multi_head_attention_9/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*K
shared_name<:Adam/transformer_block_9/multi_head_attention_9/key/bias/v
 
NAdam/transformer_block_9/multi_head_attention_9/key/bias/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_block_9/multi_head_attention_9/key/bias/v*
_output_shapes
:	А*
dtype0
ё
>Adam/transformer_block_9/multi_head_attention_9/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*O
shared_name@>Adam/transformer_block_9/multi_head_attention_9/value/kernel/v
„
RAdam/transformer_block_9/multi_head_attention_9/value/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_9/multi_head_attention_9/value/kernel/v*$
_output_shapes
:АА*
dtype0
’
<Adam/transformer_block_9/multi_head_attention_9/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*M
shared_name><Adam/transformer_block_9/multi_head_attention_9/value/bias/v
ќ
PAdam/transformer_block_9/multi_head_attention_9/value/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_9/multi_head_attention_9/value/bias/v*
_output_shapes
:	А*
dtype0
ф
IAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*Z
shared_nameKIAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/v
н
]Adam/transformer_block_9/multi_head_attention_9/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/v*$
_output_shapes
:АА*
dtype0
з
GAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*X
shared_nameIGAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/v
а
[Adam/transformer_block_9/multi_head_attention_9/attention_output/bias/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_24/kernel/v
Г
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_24/bias/v
z
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_25/kernel/v
Г
*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v* 
_output_shapes
:
АА*
dtype0
Б
Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_25/bias/v
z
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes	
:А*
dtype0
«
7Adam/transformer_block_9/layer_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*H
shared_name97Adam/transformer_block_9/layer_normalization_18/gamma/v
ј
KAdam/transformer_block_9/layer_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_9/layer_normalization_18/gamma/v*
_output_shapes	
:А*
dtype0
≈
6Adam/transformer_block_9/layer_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*G
shared_name86Adam/transformer_block_9/layer_normalization_18/beta/v
Њ
JAdam/transformer_block_9/layer_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_9/layer_normalization_18/beta/v*
_output_shapes	
:А*
dtype0
«
7Adam/transformer_block_9/layer_normalization_19/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*H
shared_name97Adam/transformer_block_9/layer_normalization_19/gamma/v
ј
KAdam/transformer_block_9/layer_normalization_19/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_9/layer_normalization_19/gamma/v*
_output_shapes	
:А*
dtype0
≈
6Adam/transformer_block_9/layer_normalization_19/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*G
shared_name86Adam/transformer_block_9/layer_normalization_19/beta/v
Њ
JAdam/transformer_block_9/layer_normalization_19/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_9/layer_normalization_19/beta/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
ґ±
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*р∞
valueе∞Bб∞ Bў∞
є
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
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
n
	token_emb
pos_emb
trainable_variables
regularization_losses
	variables
	keras_api
Ч
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
 regularization_losses
!	variables
"	keras_api
R
#trainable_variables
$regularization_losses
%	variables
&	keras_api
†
'att
(ffn
)
layernorm1
*
layernorm2
+dropout1
,dropout2
-trainable_variables
.regularization_losses
/	variables
0	keras_api
R
1trainable_variables
2regularization_losses
3	variables
4	keras_api
 
R
5trainable_variables
6regularization_losses
7	variables
8	keras_api
h

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
R
?trainable_variables
@regularization_losses
A	variables
B	keras_api
h

Ckernel
Dbias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
R
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
h

Mkernel
Nbias
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
»

Sbeta_1

Tbeta_2
	Udecay
Vlearning_rate
WitermЌmќ9mѕ:m–Cm—Dm“Mm”Nm‘Xm’Ym÷Zm„[mЎ\mў]mЏ^mџ_m№`mЁamёbmяcmаdmбemвfmгgmдhmеimжvзvи9vй:vкCvлDvмMvнNvоXvпYvрZvс[vт\vу]vф^vх_vц`vчavшbvщcvъdvыevьfvэgvюhv€ivА
∆
X0
Y1
2
3
Z4
[5
\6
]7
^8
_9
`10
a11
b12
c13
d14
e15
f16
g17
h18
i19
920
:21
C22
D23
M24
N25
 
÷
X0
Y1
2
3
4
5
Z6
[7
\8
]9
^10
_11
`12
a13
b14
c15
d16
e17
f18
g19
h20
i21
922
:23
C24
D25
M26
N27
≠
jnon_trainable_variables
trainable_variables

klayers
regularization_losses
	variables
llayer_metrics
mmetrics
nlayer_regularization_losses
 
b
X
embeddings
otrainable_variables
pregularization_losses
q	variables
r	keras_api
b
Y
embeddings
strainable_variables
tregularization_losses
u	variables
v	keras_api

X0
Y1
 

X0
Y1
≠
wnon_trainable_variables
trainable_variables

xlayers
regularization_losses
ylayer_metrics
	variables
zlayer_regularization_losses
{metrics
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
2
3
Ѓ
|non_trainable_variables
trainable_variables

}layers
 regularization_losses
~layer_metrics
!	variables
layer_regularization_losses
Аmetrics
 
 
 
≤
Бnon_trainable_variables
#trainable_variables
Вlayers
$regularization_losses
Гlayer_metrics
%	variables
 Дlayer_regularization_losses
Еmetrics
≈
Ж_query_dense
З
_key_dense
И_value_dense
Й_softmax
К_dropout_layer
Л_output_dense
Мtrainable_variables
Нregularization_losses
О	variables
П	keras_api
®
Рlayer_with_weights-0
Рlayer-0
Сlayer_with_weights-1
Сlayer-1
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
v
	Цaxis
	fgamma
gbeta
Чtrainable_variables
Шregularization_losses
Щ	variables
Ъ	keras_api
v
	Ыaxis
	hgamma
ibeta
Ьtrainable_variables
Эregularization_losses
Ю	variables
Я	keras_api
V
†trainable_variables
°regularization_losses
Ґ	variables
£	keras_api
V
§trainable_variables
•regularization_losses
¶	variables
І	keras_api
v
Z0
[1
\2
]3
^4
_5
`6
a7
b8
c9
d10
e11
f12
g13
h14
i15
 
v
Z0
[1
\2
]3
^4
_5
`6
a7
b8
c9
d10
e11
f12
g13
h14
i15
≤
®non_trainable_variables
-trainable_variables
©layers
.regularization_losses
™layer_metrics
/	variables
 Ђlayer_regularization_losses
ђmetrics
 
 
 
≤
≠non_trainable_variables
1trainable_variables
Ѓlayers
2regularization_losses
ѓlayer_metrics
3	variables
 ∞layer_regularization_losses
±metrics
 
 
 
≤
≤non_trainable_variables
5trainable_variables
≥layers
6regularization_losses
іlayer_metrics
7	variables
 µlayer_regularization_losses
ґmetrics
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
≤
Јnon_trainable_variables
;trainable_variables
Єlayers
<regularization_losses
єlayer_metrics
=	variables
 Їlayer_regularization_losses
їmetrics
 
 
 
≤
Љnon_trainable_variables
?trainable_variables
љlayers
@regularization_losses
Њlayer_metrics
A	variables
 њlayer_regularization_losses
јmetrics
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
≤
Ѕnon_trainable_variables
Etrainable_variables
¬layers
Fregularization_losses
√layer_metrics
G	variables
 ƒlayer_regularization_losses
≈metrics
 
 
 
≤
∆non_trainable_variables
Itrainable_variables
«layers
Jregularization_losses
»layer_metrics
K	variables
 …layer_regularization_losses
 metrics
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
 

M0
N1
≤
Ћnon_trainable_variables
Otrainable_variables
ћlayers
Pregularization_losses
Ќlayer_metrics
Q	variables
 ќlayer_regularization_losses
ѕmetrics
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
{y
VARIABLE_VALUE5token_and_position_embedding_4/embedding_8/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5token_and_position_embedding_4/embedding_9/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE7transformer_block_9/multi_head_attention_9/query/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5transformer_block_9/multi_head_attention_9/query/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5transformer_block_9/multi_head_attention_9/key/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE3transformer_block_9/multi_head_attention_9/key/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE7transformer_block_9/multi_head_attention_9/value/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5transformer_block_9/multi_head_attention_9/value/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUEBtransformer_block_9/multi_head_attention_9/attention_output/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE@transformer_block_9/multi_head_attention_9/attention_output/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_24/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_24/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_25/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_25/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_9/layer_normalization_18/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_9/layer_normalization_18/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_9/layer_normalization_19/gamma1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_9/layer_normalization_19/beta1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE

0
1
^
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
 

–0
 

X0
 

X0
≤
—non_trainable_variables
otrainable_variables
“layers
pregularization_losses
”layer_metrics
q	variables
 ‘layer_regularization_losses
’metrics

Y0
 

Y0
≤
÷non_trainable_variables
strainable_variables
„layers
tregularization_losses
Ўlayer_metrics
u	variables
 ўlayer_regularization_losses
Џmetrics
 

0
1
 
 
 

0
1
 
 
 
 
 
 
 
 
 
Я
џpartial_output_shape
№full_output_shape

Zkernel
[bias
Ёtrainable_variables
ёregularization_losses
я	variables
а	keras_api
Я
бpartial_output_shape
вfull_output_shape

\kernel
]bias
гtrainable_variables
дregularization_losses
е	variables
ж	keras_api
Я
зpartial_output_shape
иfull_output_shape

^kernel
_bias
йtrainable_variables
кregularization_losses
л	variables
м	keras_api
V
нtrainable_variables
оregularization_losses
п	variables
р	keras_api
V
сtrainable_variables
тregularization_losses
у	variables
ф	keras_api
Я
хpartial_output_shape
цfull_output_shape

`kernel
abias
чtrainable_variables
шregularization_losses
щ	variables
ъ	keras_api
8
Z0
[1
\2
]3
^4
_5
`6
a7
 
8
Z0
[1
\2
]3
^4
_5
`6
a7
µ
ыnon_trainable_variables
Мtrainable_variables
ьlayers
Нregularization_losses
эlayer_metrics
О	variables
 юlayer_regularization_losses
€metrics
l

bkernel
cbias
Аtrainable_variables
Бregularization_losses
В	variables
Г	keras_api
l

dkernel
ebias
Дtrainable_variables
Еregularization_losses
Ж	variables
З	keras_api

b0
c1
d2
e3
 

b0
c1
d2
e3
µ
Иnon_trainable_variables
Тtrainable_variables
Йlayers
Уregularization_losses
Ф	variables
Кlayer_metrics
Лmetrics
 Мlayer_regularization_losses
 

f0
g1
 

f0
g1
µ
Нnon_trainable_variables
Чtrainable_variables
Оlayers
Шregularization_losses
Пlayer_metrics
Щ	variables
 Рlayer_regularization_losses
Сmetrics
 

h0
i1
 

h0
i1
µ
Тnon_trainable_variables
Ьtrainable_variables
Уlayers
Эregularization_losses
Фlayer_metrics
Ю	variables
 Хlayer_regularization_losses
Цmetrics
 
 
 
µ
Чnon_trainable_variables
†trainable_variables
Шlayers
°regularization_losses
Щlayer_metrics
Ґ	variables
 Ъlayer_regularization_losses
Ыmetrics
 
 
 
µ
Ьnon_trainable_variables
§trainable_variables
Эlayers
•regularization_losses
Юlayer_metrics
¶	variables
 Яlayer_regularization_losses
†metrics
 
*
'0
(1
)2
*3
+4
,5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

°total

Ґcount
£	variables
§	keras_api
 
 
 
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
Z0
[1
 

Z0
[1
µ
•non_trainable_variables
Ёtrainable_variables
¶layers
ёregularization_losses
Іlayer_metrics
я	variables
 ®layer_regularization_losses
©metrics
 
 

\0
]1
 

\0
]1
µ
™non_trainable_variables
гtrainable_variables
Ђlayers
дregularization_losses
ђlayer_metrics
е	variables
 ≠layer_regularization_losses
Ѓmetrics
 
 

^0
_1
 

^0
_1
µ
ѓnon_trainable_variables
йtrainable_variables
∞layers
кregularization_losses
±layer_metrics
л	variables
 ≤layer_regularization_losses
≥metrics
 
 
 
µ
іnon_trainable_variables
нtrainable_variables
µlayers
оregularization_losses
ґlayer_metrics
п	variables
 Јlayer_regularization_losses
Єmetrics
 
 
 
µ
єnon_trainable_variables
сtrainable_variables
Їlayers
тregularization_losses
їlayer_metrics
у	variables
 Љlayer_regularization_losses
љmetrics
 
 

`0
a1
 

`0
a1
µ
Њnon_trainable_variables
чtrainable_variables
њlayers
шregularization_losses
јlayer_metrics
щ	variables
 Ѕlayer_regularization_losses
¬metrics
 
0
Ж0
З1
И2
Й3
К4
Л5
 
 
 

b0
c1
 

b0
c1
µ
√non_trainable_variables
Аtrainable_variables
ƒlayers
Бregularization_losses
≈layer_metrics
В	variables
 ∆layer_regularization_losses
«metrics

d0
e1
 

d0
e1
µ
»non_trainable_variables
Дtrainable_variables
…layers
Еregularization_losses
 layer_metrics
Ж	variables
 Ћlayer_regularization_losses
ћmetrics
 

Р0
С1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
°0
Ґ1

£	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
КЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE<Adam/token_and_position_embedding_4/embedding_8/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE<Adam/token_and_position_embedding_4/embedding_9/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
°Ю
VARIABLE_VALUE>Adam/transformer_block_9/multi_head_attention_9/query/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE<Adam/transformer_block_9/multi_head_attention_9/query/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE<Adam/transformer_block_9/multi_head_attention_9/key/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЭЪ
VARIABLE_VALUE:Adam/transformer_block_9/multi_head_attention_9/key/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
°Ю
VARIABLE_VALUE>Adam/transformer_block_9/multi_head_attention_9/value/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE<Adam/transformer_block_9/multi_head_attention_9/value/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
≠™
VARIABLE_VALUEIAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ђ®
VARIABLE_VALUEGAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_24/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_24/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_25/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_25/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЫШ
VARIABLE_VALUE7Adam/transformer_block_9/layer_normalization_18/gamma/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE6Adam/transformer_block_9/layer_normalization_18/beta/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЫШ
VARIABLE_VALUE7Adam/transformer_block_9/layer_normalization_19/gamma/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE6Adam/transformer_block_9/layer_normalization_19/beta/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE<Adam/token_and_position_embedding_4/embedding_8/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE<Adam/token_and_position_embedding_4/embedding_9/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
°Ю
VARIABLE_VALUE>Adam/transformer_block_9/multi_head_attention_9/query/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE<Adam/transformer_block_9/multi_head_attention_9/query/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE<Adam/transformer_block_9/multi_head_attention_9/key/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЭЪ
VARIABLE_VALUE:Adam/transformer_block_9/multi_head_attention_9/key/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
°Ю
VARIABLE_VALUE>Adam/transformer_block_9/multi_head_attention_9/value/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЯЬ
VARIABLE_VALUE<Adam/transformer_block_9/multi_head_attention_9/value/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
≠™
VARIABLE_VALUEIAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ђ®
VARIABLE_VALUEGAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_24/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_24/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_25/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_25/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЫШ
VARIABLE_VALUE7Adam/transformer_block_9/layer_normalization_18/gamma/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE6Adam/transformer_block_9/layer_normalization_18/beta/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЫШ
VARIABLE_VALUE7Adam/transformer_block_9/layer_normalization_19/gamma/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE6Adam/transformer_block_9/layer_normalization_19/beta/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_input_10Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
|
serving_default_input_9Placeholder*(
_output_shapes
:€€€€€€€€€ДR*
dtype0*
shape:€€€€€€€€€ДR
∞
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10serving_default_input_95token_and_position_embedding_4/embedding_9/embeddings5token_and_position_embedding_4/embedding_8/embeddings%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/beta7transformer_block_9/multi_head_attention_9/query/kernel5transformer_block_9/multi_head_attention_9/query/bias5transformer_block_9/multi_head_attention_9/key/kernel3transformer_block_9/multi_head_attention_9/key/bias7transformer_block_9/multi_head_attention_9/value/kernel5transformer_block_9/multi_head_attention_9/value/biasBtransformer_block_9/multi_head_attention_9/attention_output/kernel@transformer_block_9/multi_head_attention_9/attention_output/bias0transformer_block_9/layer_normalization_18/gamma/transformer_block_9/layer_normalization_18/betadense_24/kerneldense_24/biasdense_25/kerneldense_25/bias0transformer_block_9/layer_normalization_19/gamma/transformer_block_9/layer_normalization_19/betadense_26/kerneldense_26/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/bias*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_44095
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
“+
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpItoken_and_position_embedding_4/embedding_8/embeddings/Read/ReadVariableOpItoken_and_position_embedding_4/embedding_9/embeddings/Read/ReadVariableOpKtransformer_block_9/multi_head_attention_9/query/kernel/Read/ReadVariableOpItransformer_block_9/multi_head_attention_9/query/bias/Read/ReadVariableOpItransformer_block_9/multi_head_attention_9/key/kernel/Read/ReadVariableOpGtransformer_block_9/multi_head_attention_9/key/bias/Read/ReadVariableOpKtransformer_block_9/multi_head_attention_9/value/kernel/Read/ReadVariableOpItransformer_block_9/multi_head_attention_9/value/bias/Read/ReadVariableOpVtransformer_block_9/multi_head_attention_9/attention_output/kernel/Read/ReadVariableOpTtransformer_block_9/multi_head_attention_9/attention_output/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOpDtransformer_block_9/layer_normalization_18/gamma/Read/ReadVariableOpCtransformer_block_9/layer_normalization_18/beta/Read/ReadVariableOpDtransformer_block_9/layer_normalization_19/gamma/Read/ReadVariableOpCtransformer_block_9/layer_normalization_19/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOpPAdam/token_and_position_embedding_4/embedding_8/embeddings/m/Read/ReadVariableOpPAdam/token_and_position_embedding_4/embedding_9/embeddings/m/Read/ReadVariableOpRAdam/transformer_block_9/multi_head_attention_9/query/kernel/m/Read/ReadVariableOpPAdam/transformer_block_9/multi_head_attention_9/query/bias/m/Read/ReadVariableOpPAdam/transformer_block_9/multi_head_attention_9/key/kernel/m/Read/ReadVariableOpNAdam/transformer_block_9/multi_head_attention_9/key/bias/m/Read/ReadVariableOpRAdam/transformer_block_9/multi_head_attention_9/value/kernel/m/Read/ReadVariableOpPAdam/transformer_block_9/multi_head_attention_9/value/bias/m/Read/ReadVariableOp]Adam/transformer_block_9/multi_head_attention_9/attention_output/kernel/m/Read/ReadVariableOp[Adam/transformer_block_9/multi_head_attention_9/attention_output/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOpKAdam/transformer_block_9/layer_normalization_18/gamma/m/Read/ReadVariableOpJAdam/transformer_block_9/layer_normalization_18/beta/m/Read/ReadVariableOpKAdam/transformer_block_9/layer_normalization_19/gamma/m/Read/ReadVariableOpJAdam/transformer_block_9/layer_normalization_19/beta/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOpPAdam/token_and_position_embedding_4/embedding_8/embeddings/v/Read/ReadVariableOpPAdam/token_and_position_embedding_4/embedding_9/embeddings/v/Read/ReadVariableOpRAdam/transformer_block_9/multi_head_attention_9/query/kernel/v/Read/ReadVariableOpPAdam/transformer_block_9/multi_head_attention_9/query/bias/v/Read/ReadVariableOpPAdam/transformer_block_9/multi_head_attention_9/key/kernel/v/Read/ReadVariableOpNAdam/transformer_block_9/multi_head_attention_9/key/bias/v/Read/ReadVariableOpRAdam/transformer_block_9/multi_head_attention_9/value/kernel/v/Read/ReadVariableOpPAdam/transformer_block_9/multi_head_attention_9/value/bias/v/Read/ReadVariableOp]Adam/transformer_block_9/multi_head_attention_9/attention_output/kernel/v/Read/ReadVariableOp[Adam/transformer_block_9/multi_head_attention_9/attention_output/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOpKAdam/transformer_block_9/layer_normalization_18/gamma/v/Read/ReadVariableOpJAdam/transformer_block_9/layer_normalization_18/beta/v/Read/ReadVariableOpKAdam/transformer_block_9/layer_normalization_19/gamma/v/Read/ReadVariableOpJAdam/transformer_block_9/layer_normalization_19/beta/v/Read/ReadVariableOpConst*d
Tin]
[2Y	*
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
GPU2*0J 8В *'
f"R 
__inference__traced_save_45845
Б
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_26/kerneldense_26/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasbeta_1beta_2decaylearning_rate	Adam/iter5token_and_position_embedding_4/embedding_8/embeddings5token_and_position_embedding_4/embedding_9/embeddings7transformer_block_9/multi_head_attention_9/query/kernel5transformer_block_9/multi_head_attention_9/query/bias5transformer_block_9/multi_head_attention_9/key/kernel3transformer_block_9/multi_head_attention_9/key/bias7transformer_block_9/multi_head_attention_9/value/kernel5transformer_block_9/multi_head_attention_9/value/biasBtransformer_block_9/multi_head_attention_9/attention_output/kernel@transformer_block_9/multi_head_attention_9/attention_output/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/bias0transformer_block_9/layer_normalization_18/gamma/transformer_block_9/layer_normalization_18/beta0transformer_block_9/layer_normalization_19/gamma/transformer_block_9/layer_normalization_19/betatotalcount"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/dense_26/kernel/mAdam/dense_26/bias/mAdam/dense_27/kernel/mAdam/dense_27/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/m<Adam/token_and_position_embedding_4/embedding_8/embeddings/m<Adam/token_and_position_embedding_4/embedding_9/embeddings/m>Adam/transformer_block_9/multi_head_attention_9/query/kernel/m<Adam/transformer_block_9/multi_head_attention_9/query/bias/m<Adam/transformer_block_9/multi_head_attention_9/key/kernel/m:Adam/transformer_block_9/multi_head_attention_9/key/bias/m>Adam/transformer_block_9/multi_head_attention_9/value/kernel/m<Adam/transformer_block_9/multi_head_attention_9/value/bias/mIAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/mGAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_25/kernel/mAdam/dense_25/bias/m7Adam/transformer_block_9/layer_normalization_18/gamma/m6Adam/transformer_block_9/layer_normalization_18/beta/m7Adam/transformer_block_9/layer_normalization_19/gamma/m6Adam/transformer_block_9/layer_normalization_19/beta/m"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/dense_26/kernel/vAdam/dense_26/bias/vAdam/dense_27/kernel/vAdam/dense_27/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/v<Adam/token_and_position_embedding_4/embedding_8/embeddings/v<Adam/token_and_position_embedding_4/embedding_9/embeddings/v>Adam/transformer_block_9/multi_head_attention_9/query/kernel/v<Adam/transformer_block_9/multi_head_attention_9/query/bias/v<Adam/transformer_block_9/multi_head_attention_9/key/kernel/v:Adam/transformer_block_9/multi_head_attention_9/key/bias/v>Adam/transformer_block_9/multi_head_attention_9/value/kernel/v<Adam/transformer_block_9/multi_head_attention_9/value/bias/vIAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/vGAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/vAdam/dense_25/kernel/vAdam/dense_25/bias/v7Adam/transformer_block_9/layer_normalization_18/gamma/v6Adam/transformer_block_9/layer_normalization_18/beta/v7Adam/transformer_block_9/layer_normalization_19/gamma/v6Adam/transformer_block_9/layer_normalization_19/beta/v*c
Tin\
Z2X*
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
GPU2*0J 8В **
f%R#
!__inference__traced_restore_46116Ї„ 
ъ
З
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43090

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
batchnorm/add_1б
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:€€€€€€€€€ДRА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):€€€€€€€€€ДRА::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€ДRА
 
_user_specified_nameinputs
Ё
}
(__inference_dense_28_layer_call_fn_45341

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_436652
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
€<
ѓ	
B__inference_model_2_layer_call_and_return_conditional_losses_43964

inputs
inputs_1(
$token_and_position_embedding_4_43896(
$token_and_position_embedding_4_43898
batch_normalization_4_43901
batch_normalization_4_43903
batch_normalization_4_43905
batch_normalization_4_43907
transformer_block_9_43911
transformer_block_9_43913
transformer_block_9_43915
transformer_block_9_43917
transformer_block_9_43919
transformer_block_9_43921
transformer_block_9_43923
transformer_block_9_43925
transformer_block_9_43927
transformer_block_9_43929
transformer_block_9_43931
transformer_block_9_43933
transformer_block_9_43935
transformer_block_9_43937
transformer_block_9_43939
transformer_block_9_43941
dense_26_43946
dense_26_43948
dense_27_43952
dense_27_43954
dense_28_43958
dense_28_43960
identityИҐ-batch_normalization_4/StatefulPartitionedCallҐ dense_26/StatefulPartitionedCallҐ dense_27/StatefulPartitionedCallҐ dense_28/StatefulPartitionedCallҐ6token_and_position_embedding_4/StatefulPartitionedCallҐ+transformer_block_9/StatefulPartitionedCallИ
6token_and_position_embedding_4/StatefulPartitionedCallStatefulPartitionedCallinputs$token_and_position_embedding_4_43896$token_and_position_embedding_4_43898*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_4301928
6token_and_position_embedding_4/StatefulPartitionedCall“
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0batch_normalization_4_43901batch_normalization_4_43903batch_normalization_4_43905batch_normalization_4_43907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_430902/
-batch_normalization_4/StatefulPartitionedCallђ
#average_pooling1d_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_428172%
#average_pooling1d_4/PartitionedCallМ
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0transformer_block_9_43911transformer_block_9_43913transformer_block_9_43915transformer_block_9_43917transformer_block_9_43919transformer_block_9_43921transformer_block_9_43923transformer_block_9_43925transformer_block_9_43927transformer_block_9_43929transformer_block_9_43931transformer_block_9_43933transformer_block_9_43935transformer_block_9_43937transformer_block_9_43939transformer_block_9_43941*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_434022-
+transformer_block_9/StatefulPartitionedCallИ
flatten_2/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_435172
flatten_2/PartitionedCallН
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_435322
concatenate_2/PartitionedCallі
 dense_26/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_26_43946dense_26_43948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_435522"
 dense_26/StatefulPartitionedCall€
dropout_24/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_435852
dropout_24/PartitionedCall±
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0dense_27_43952dense_27_43954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_436092"
 dense_27/StatefulPartitionedCall€
dropout_25/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_436422
dropout_25/PartitionedCall±
 dense_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_28_43958dense_28_43960*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_436652"
 dense_28/StatefulPartitionedCallэ
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall7^token_and_position_embedding_4/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2p
6token_and_position_embedding_4/StatefulPartitionedCall6token_and_position_embedding_4/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ДR
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
С	
№
C__inference_dense_28_layer_call_and_return_conditional_losses_43665

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ь0
≈
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_42764

inputs
assignmovingavg_42739
assignmovingavg_1_42745)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient≤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЈ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/42739*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42739*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/42739*
_output_shapes	
:А2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/42739*
_output_shapes	
:А2
AssignMovingAvg/mul≠
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42739AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/42739*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp—
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/42745*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42745*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/42745*
_output_shapes	
:А2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/42745*
_output_shapes	
:А2
AssignMovingAvg_1/mulє
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42745AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/42745*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
batchnorm/add_1Ѕ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:€€€€€€€€€€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
р	
№
C__inference_dense_26_layer_call_and_return_conditional_losses_45239

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	И@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€И::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€И
 
_user_specified_nameinputs
џ
в
C__inference_dense_25_layer_call_and_return_conditional_losses_42904

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpШ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
АА*
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЯ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ћ0
≈
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43070

inputs
assignmovingavg_43045
assignmovingavg_1_43051)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient™
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЈ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/43045*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_43045*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/43045*
_output_shapes	
:А2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/43045*
_output_shapes	
:А2
AssignMovingAvg/mul≠
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_43045AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/43045*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp—
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/43051*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_43051*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/43051*
_output_shapes	
:А2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/43051*
_output_shapes	
:А2
AssignMovingAvg_1/mulє
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_43051AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/43051*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
batchnorm/add_1є
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:€€€€€€€€€ДRА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):€€€€€€€€€ДRА::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€ДRА
 
_user_specified_nameinputs
о
®
5__inference_batch_normalization_4_layer_call_fn_44760

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_427642
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
≈
Ґ
'__inference_model_2_layer_call_fn_44023
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

unknown_26
identityИҐStatefulPartitionedCallд
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
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_439642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_9:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_10
у
Б
Y__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_43019
x&
"embedding_9_embedding_lookup_43006&
"embedding_8_embedding_lookup_43012
identityИҐembedding_8/embedding_lookupҐembedding_9/embedding_lookup?
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
€€€€€€€€€2
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
strided_slice/stack_2в
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
range/deltaА
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:€€€€€€€€€2
rangeЃ
embedding_9/embedding_lookupResourceGather"embedding_9_embedding_lookup_43006range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_9/embedding_lookup/43006*(
_output_shapes
:€€€€€€€€€А*
dtype02
embedding_9/embedding_lookupЩ
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_9/embedding_lookup/43006*(
_output_shapes
:€€€€€€€€€А2'
%embedding_9/embedding_lookup/IdentityЅ
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2)
'embedding_9/embedding_lookup/Identity_1q
embedding_8/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€ДR2
embedding_8/Castє
embedding_8/embedding_lookupResourceGather"embedding_8_embedding_lookup_43012embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_8/embedding_lookup/43012*-
_output_shapes
:€€€€€€€€€ДRА*
dtype02
embedding_8/embedding_lookupЮ
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_8/embedding_lookup/43012*-
_output_shapes
:€€€€€€€€€ДRА2'
%embedding_8/embedding_lookup/Identity∆
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2)
'embedding_8/embedding_lookup/Identity_1ѓ
addAddV20embedding_8/embedding_lookup/Identity_1:output:00embedding_9/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
addЯ
IdentityIdentityadd:z:0^embedding_8/embedding_lookup^embedding_9/embedding_lookup*
T0*-
_output_shapes
:€€€€€€€€€ДRА2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€ДR::2<
embedding_8/embedding_lookupembedding_8/embedding_lookup2<
embedding_9/embedding_lookupembedding_9/embedding_lookup:K G
(
_output_shapes
:€€€€€€€€€ДR

_user_specified_namex
ъ
З
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44829

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
batchnorm/add_1б
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:€€€€€€€€€ДRА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):€€€€€€€€€ДRА::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€ДRА
 
_user_specified_nameinputs
Ѕ
t
H__inference_concatenate_2_layer_call_and_return_conditional_losses_45222
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisВ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€И2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€И2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€А:€€€€€€€€€:R N
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
Н
d
E__inference_dropout_24_layer_call_and_return_conditional_losses_45260

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeј
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
†J
Ѓ
G__inference_sequential_9_layer_call_and_return_conditional_losses_45398

inputs.
*dense_24_tensordot_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource.
*dense_25_tensordot_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource
identityИҐdense_24/BiasAdd/ReadVariableOpҐ!dense_24/Tensordot/ReadVariableOpҐdense_25/BiasAdd/ReadVariableOpҐ!dense_25/Tensordot/ReadVariableOp≥
!dense_24/Tensordot/ReadVariableOpReadVariableOp*dense_24_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!dense_24/Tensordot/ReadVariableOp|
dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_24/Tensordot/axesГ
dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_24/Tensordot/freej
dense_24/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_24/Tensordot/ShapeЖ
 dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/GatherV2/axisю
dense_24/Tensordot/GatherV2GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/free:output:0)dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2К
"dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_24/Tensordot/GatherV2_1/axisД
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
dense_24/Tensordot/Const§
dense_24/Tensordot/ProdProd$dense_24/Tensordot/GatherV2:output:0!dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/ProdВ
dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_1ђ
dense_24/Tensordot/Prod_1Prod&dense_24/Tensordot/GatherV2_1:output:0#dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod_1В
dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_24/Tensordot/concat/axisЁ
dense_24/Tensordot/concatConcatV2 dense_24/Tensordot/free:output:0 dense_24/Tensordot/axes:output:0'dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat∞
dense_24/Tensordot/stackPack dense_24/Tensordot/Prod:output:0"dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/stackђ
dense_24/Tensordot/transpose	Transposeinputs"dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_24/Tensordot/transpose√
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_24/Tensordot/Reshape√
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_24/Tensordot/MatMulГ
dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_24/Tensordot/Const_2Ж
 dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/concat_1/axisк
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat_1µ
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_24/Tensordot®
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_24/BiasAdd/ReadVariableOpђ
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_24/BiasAddx
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_24/Relu≥
!dense_25/Tensordot/ReadVariableOpReadVariableOp*dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!dense_25/Tensordot/ReadVariableOp|
dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_25/Tensordot/axesГ
dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_25/Tensordot/free
dense_25/Tensordot/ShapeShapedense_24/Relu:activations:0*
T0*
_output_shapes
:2
dense_25/Tensordot/ShapeЖ
 dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/GatherV2/axisю
dense_25/Tensordot/GatherV2GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/free:output:0)dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2К
"dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_25/Tensordot/GatherV2_1/axisД
dense_25/Tensordot/GatherV2_1GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/axes:output:0+dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2_1~
dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const§
dense_25/Tensordot/ProdProd$dense_25/Tensordot/GatherV2:output:0!dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/ProdВ
dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const_1ђ
dense_25/Tensordot/Prod_1Prod&dense_25/Tensordot/GatherV2_1:output:0#dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/Prod_1В
dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_25/Tensordot/concat/axisЁ
dense_25/Tensordot/concatConcatV2 dense_25/Tensordot/free:output:0 dense_25/Tensordot/axes:output:0'dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat∞
dense_25/Tensordot/stackPack dense_25/Tensordot/Prod:output:0"dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/stackЅ
dense_25/Tensordot/transpose	Transposedense_24/Relu:activations:0"dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_25/Tensordot/transpose√
dense_25/Tensordot/ReshapeReshape dense_25/Tensordot/transpose:y:0!dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_25/Tensordot/Reshape√
dense_25/Tensordot/MatMulMatMul#dense_25/Tensordot/Reshape:output:0)dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_25/Tensordot/MatMulГ
dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_25/Tensordot/Const_2Ж
 dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/concat_1/axisк
dense_25/Tensordot/concat_1ConcatV2$dense_25/Tensordot/GatherV2:output:0#dense_25/Tensordot/Const_2:output:0)dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat_1µ
dense_25/TensordotReshape#dense_25/Tensordot/MatMul:product:0$dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_25/Tensordot®
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_25/BiasAdd/ReadVariableOpђ
dense_25/BiasAddBiasAdddense_25/Tensordot:output:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_25/BiasAddю
IdentityIdentitydense_25/BiasAdd:output:0 ^dense_24/BiasAdd/ReadVariableOp"^dense_24/Tensordot/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp"^dense_25/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€А::::2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2F
!dense_24/Tensordot/ReadVariableOp!dense_24/Tensordot/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2F
!dense_25/Tensordot/ReadVariableOp!dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ї
Я
,__inference_sequential_9_layer_call_fn_45468

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_429522
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
“

я
3__inference_transformer_block_9_layer_call_fn_45167

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
identityИҐStatefulPartitionedCallЅ
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
:€€€€€€€€€А*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_432752
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€А::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
н	
№
C__inference_dense_27_layer_call_and_return_conditional_losses_45286

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
°
E
)__inference_flatten_2_layer_call_fn_45215

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_435172
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€А:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
√
Ґ
'__inference_model_2_layer_call_fn_43889
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

unknown_26
identityИҐStatefulPartitionedCallв
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
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_438302
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_9:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_10
ВС
С
B__inference_model_2_layer_call_and_return_conditional_losses_44534
inputs_0
inputs_1E
Atoken_and_position_embedding_4_embedding_9_embedding_lookup_44352E
Atoken_and_position_embedding_4_embedding_8_embedding_lookup_44358;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resourceZ
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
Ktransformer_block_9_sequential_9_dense_24_tensordot_readvariableop_resourceM
Itransformer_block_9_sequential_9_dense_24_biasadd_readvariableop_resourceO
Ktransformer_block_9_sequential_9_dense_25_tensordot_readvariableop_resourceM
Itransformer_block_9_sequential_9_dense_25_biasadd_readvariableop_resourceT
Ptransformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource
identityИҐ.batch_normalization_4/batchnorm/ReadVariableOpҐ0batch_normalization_4/batchnorm/ReadVariableOp_1Ґ0batch_normalization_4/batchnorm/ReadVariableOp_2Ґ2batch_normalization_4/batchnorm/mul/ReadVariableOpҐdense_26/BiasAdd/ReadVariableOpҐdense_26/MatMul/ReadVariableOpҐdense_27/BiasAdd/ReadVariableOpҐdense_27/MatMul/ReadVariableOpҐdense_28/BiasAdd/ReadVariableOpҐdense_28/MatMul/ReadVariableOpҐ;token_and_position_embedding_4/embedding_8/embedding_lookupҐ;token_and_position_embedding_4/embedding_9/embedding_lookupҐCtransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpҐGtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpҐCtransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpҐGtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpҐNtransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpҐXtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpҐAtransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpҐKtransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpҐCtransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpҐMtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpҐCtransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpҐMtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpҐ@transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOpҐBtransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpҐ@transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOpҐBtransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpД
$token_and_position_embedding_4/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_4/Shapeї
2token_and_position_embedding_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€24
2token_and_position_embedding_4/strided_slice/stackґ
4token_and_position_embedding_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_4/strided_slice/stack_1ґ
4token_and_position_embedding_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_4/strided_slice/stack_2Ь
,token_and_position_embedding_4/strided_sliceStridedSlice-token_and_position_embedding_4/Shape:output:0;token_and_position_embedding_4/strided_slice/stack:output:0=token_and_position_embedding_4/strided_slice/stack_1:output:0=token_and_position_embedding_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_4/strided_sliceЪ
*token_and_position_embedding_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_4/range/startЪ
*token_and_position_embedding_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_4/range/deltaЫ
$token_and_position_embedding_4/rangeRange3token_and_position_embedding_4/range/start:output:05token_and_position_embedding_4/strided_slice:output:03token_and_position_embedding_4/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2&
$token_and_position_embedding_4/range…
;token_and_position_embedding_4/embedding_9/embedding_lookupResourceGatherAtoken_and_position_embedding_4_embedding_9_embedding_lookup_44352-token_and_position_embedding_4/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_4/embedding_9/embedding_lookup/44352*(
_output_shapes
:€€€€€€€€€А*
dtype02=
;token_and_position_embedding_4/embedding_9/embedding_lookupХ
Dtoken_and_position_embedding_4/embedding_9/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_4/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_4/embedding_9/embedding_lookup/44352*(
_output_shapes
:€€€€€€€€€А2F
Dtoken_and_position_embedding_4/embedding_9/embedding_lookup/IdentityЮ
Ftoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2H
Ftoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1ґ
/token_and_position_embedding_4/embedding_8/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€ДR21
/token_and_position_embedding_4/embedding_8/Cast‘
;token_and_position_embedding_4/embedding_8/embedding_lookupResourceGatherAtoken_and_position_embedding_4_embedding_8_embedding_lookup_443583token_and_position_embedding_4/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_4/embedding_8/embedding_lookup/44358*-
_output_shapes
:€€€€€€€€€ДRА*
dtype02=
;token_and_position_embedding_4/embedding_8/embedding_lookupЪ
Dtoken_and_position_embedding_4/embedding_8/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_4/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_4/embedding_8/embedding_lookup/44358*-
_output_shapes
:€€€€€€€€€ДRА2F
Dtoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity£
Ftoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2H
Ftoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1Ђ
"token_and_position_embedding_4/addAddV2Otoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2$
"token_and_position_embedding_4/add’
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/yб
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/add¶
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/Rsqrtб
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpё
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/mulё
%batch_normalization_4/batchnorm/mul_1Mul&token_and_position_embedding_4/add:z:0'batch_normalization_4/batchnorm/mul:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2'
%batch_normalization_4/batchnorm/mul_1џ
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1ё
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/mul_2џ
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2№
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/subг
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2'
%batch_normalization_4/batchnorm/add_1К
"average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_4/ExpandDims/dimв
average_pooling1d_4/ExpandDims
ExpandDims)batch_normalization_4/batchnorm/add_1:z:0+average_pooling1d_4/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€ДRА2 
average_pooling1d_4/ExpandDimsз
average_pooling1d_4/AvgPoolAvgPool'average_pooling1d_4/ExpandDims:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
ksize	
И'*
paddingVALID*
strides	
И'2
average_pooling1d_4/AvgPoolє
average_pooling1d_4/SqueezeSqueeze$average_pooling1d_4/AvgPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€А*
squeeze_dims
2
average_pooling1d_4/Squeezeї
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_9_multi_head_attention_9_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02O
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpи
>transformer_block_9/multi_head_attention_9/query/einsum/EinsumEinsum$average_pooling1d_4/Squeeze:output:0Utransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2@
>transformer_block_9/multi_head_attention_9/query/einsum/EinsumШ
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpReadVariableOpLtransformer_block_9_multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes
:	А*
dtype02E
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp∆
4transformer_block_9/multi_head_attention_9/query/addAddV2Gtransformer_block_9/multi_head_attention_9/query/einsum/Einsum:output:0Ktransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А26
4transformer_block_9/multi_head_attention_9/query/addµ
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_9_multi_head_attention_9_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02M
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpв
<transformer_block_9/multi_head_attention_9/key/einsum/EinsumEinsum$average_pooling1d_4/Squeeze:output:0Stransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2>
<transformer_block_9/multi_head_attention_9/key/einsum/EinsumТ
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpReadVariableOpJtransformer_block_9_multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes
:	А*
dtype02C
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpЊ
2transformer_block_9/multi_head_attention_9/key/addAddV2Etransformer_block_9/multi_head_attention_9/key/einsum/Einsum:output:0Itransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А24
2transformer_block_9/multi_head_attention_9/key/addї
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_9_multi_head_attention_9_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02O
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpи
>transformer_block_9/multi_head_attention_9/value/einsum/EinsumEinsum$average_pooling1d_4/Squeeze:output:0Utransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2@
>transformer_block_9/multi_head_attention_9/value/einsum/EinsumШ
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpReadVariableOpLtransformer_block_9_multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes
:	А*
dtype02E
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp∆
4transformer_block_9/multi_head_attention_9/value/addAddV2Gtransformer_block_9/multi_head_attention_9/value/einsum/Einsum:output:0Ktransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А26
4transformer_block_9/multi_head_attention_9/value/add©
0transformer_block_9/multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=22
0transformer_block_9/multi_head_attention_9/Mul/yЧ
.transformer_block_9/multi_head_attention_9/MulMul8transformer_block_9/multi_head_attention_9/query/add:z:09transformer_block_9/multi_head_attention_9/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А20
.transformer_block_9/multi_head_attention_9/Mulћ
8transformer_block_9/multi_head_attention_9/einsum/EinsumEinsum6transformer_block_9/multi_head_attention_9/key/add:z:02transformer_block_9/multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€*
equationaecd,abcd->acbe2:
8transformer_block_9/multi_head_attention_9/einsum/EinsumА
:transformer_block_9/multi_head_attention_9/softmax/SoftmaxSoftmaxAtransformer_block_9/multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€2<
:transformer_block_9/multi_head_attention_9/softmax/SoftmaxЖ
;transformer_block_9/multi_head_attention_9/dropout/IdentityIdentityDtransformer_block_9/multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€2=
;transformer_block_9/multi_head_attention_9/dropout/Identityе
:transformer_block_9/multi_head_attention_9/einsum_1/EinsumEinsumDtransformer_block_9/multi_head_attention_9/dropout/Identity:output:08transformer_block_9/multi_head_attention_9/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationacbe,aecd->abcd2<
:transformer_block_9/multi_head_attention_9/einsum_1/Einsum№
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_9_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02Z
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp§
Itransformer_block_9/multi_head_attention_9/attention_output/einsum/EinsumEinsumCtransformer_block_9/multi_head_attention_9/einsum_1/Einsum:output:0`transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€А*
equationabcd,cde->abe2K
Itransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsumµ
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_9_multi_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype02P
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpо
?transformer_block_9/multi_head_attention_9/attention_output/addAddV2Rtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum:output:0Vtransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2A
?transformer_block_9/multi_head_attention_9/attention_output/addЏ
'transformer_block_9/dropout_22/IdentityIdentityCtransformer_block_9/multi_head_attention_9/attention_output/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2)
'transformer_block_9/dropout_22/Identity 
transformer_block_9/addAddV2$average_pooling1d_4/Squeeze:output:00transformer_block_9/dropout_22/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
transformer_block_9/addа
Itransformer_block_9/layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_9/layer_normalization_18/moments/mean/reduction_indices≤
7transformer_block_9/layer_normalization_18/moments/meanMeantransformer_block_9/add:z:0Rtransformer_block_9/layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(29
7transformer_block_9/layer_normalization_18/moments/meanК
?transformer_block_9/layer_normalization_18/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2A
?transformer_block_9/layer_normalization_18/moments/StopGradientњ
Dtransformer_block_9/layer_normalization_18/moments/SquaredDifferenceSquaredDifferencetransformer_block_9/add:z:0Htransformer_block_9/layer_normalization_18/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2F
Dtransformer_block_9/layer_normalization_18/moments/SquaredDifferenceи
Mtransformer_block_9/layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_9/layer_normalization_18/moments/variance/reduction_indicesл
;transformer_block_9/layer_normalization_18/moments/varianceMeanHtransformer_block_9/layer_normalization_18/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2=
;transformer_block_9/layer_normalization_18/moments/varianceљ
:transformer_block_9/layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52<
:transformer_block_9/layer_normalization_18/batchnorm/add/yЊ
8transformer_block_9/layer_normalization_18/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_18/moments/variance:output:0Ctransformer_block_9/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2:
8transformer_block_9/layer_normalization_18/batchnorm/addх
:transformer_block_9/layer_normalization_18/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2<
:transformer_block_9/layer_normalization_18/batchnorm/Rsqrt†
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02I
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp√
8transformer_block_9/layer_normalization_18/batchnorm/mulMul>transformer_block_9/layer_normalization_18/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2:
8transformer_block_9/layer_normalization_18/batchnorm/mulС
:transformer_block_9/layer_normalization_18/batchnorm/mul_1Multransformer_block_9/add:z:0<transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_18/batchnorm/mul_1ґ
:transformer_block_9/layer_normalization_18/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_18/moments/mean:output:0<transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_18/batchnorm/mul_2Ф
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02E
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpњ
8transformer_block_9/layer_normalization_18/batchnorm/subSubKtransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_18/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2:
8transformer_block_9/layer_normalization_18/batchnorm/subґ
:transformer_block_9/layer_normalization_18/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_18/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_18/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_18/batchnorm/add_1Ц
Btransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_24_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype02D
Btransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpЊ
8transformer_block_9/sequential_9/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_9/sequential_9/dense_24/Tensordot/axes≈
8transformer_block_9/sequential_9/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_9/sequential_9/dense_24/Tensordot/freeд
9transformer_block_9/sequential_9/dense_24/Tensordot/ShapeShape>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_24/Tensordot/Shape»
Atransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2/axis£
<transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_24/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_24/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2ћ
Ctransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1/axis©
>transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_24/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_24/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1ј
9transformer_block_9/sequential_9/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_9/sequential_9/dense_24/Tensordot/Const®
8transformer_block_9/sequential_9/dense_24/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_9/sequential_9/dense_24/Tensordot/Prodƒ
;transformer_block_9/sequential_9/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_9/sequential_9/dense_24/Tensordot/Const_1∞
:transformer_block_9/sequential_9/dense_24/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_9/sequential_9/dense_24/Tensordot/Prod_1ƒ
?transformer_block_9/sequential_9/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_9/sequential_9/dense_24/Tensordot/concat/axisВ
:transformer_block_9/sequential_9/dense_24/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_24/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_24/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_9/sequential_9/dense_24/Tensordot/concatі
9transformer_block_9/sequential_9/dense_24/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_24/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_24/Tensordot/stack«
=transformer_block_9/sequential_9/dense_24/Tensordot/transpose	Transpose>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0Ctransformer_block_9/sequential_9/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2?
=transformer_block_9/sequential_9/dense_24/Tensordot/transpose«
;transformer_block_9/sequential_9/dense_24/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_24/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2=
;transformer_block_9/sequential_9/dense_24/Tensordot/Reshape«
:transformer_block_9/sequential_9/dense_24/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_24/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/sequential_9/dense_24/Tensordot/MatMul≈
;transformer_block_9/sequential_9/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2=
;transformer_block_9/sequential_9/dense_24/Tensordot/Const_2»
Atransformer_block_9/sequential_9/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_24/Tensordot/concat_1/axisП
<transformer_block_9/sequential_9/dense_24/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_24/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_24/Tensordot/concat_1є
3transformer_block_9/sequential_9/dense_24/TensordotReshapeDtransformer_block_9/sequential_9/dense_24/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А25
3transformer_block_9/sequential_9/dense_24/TensordotЛ
@transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02B
@transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp∞
1transformer_block_9/sequential_9/dense_24/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_24/Tensordot:output:0Htransformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А23
1transformer_block_9/sequential_9/dense_24/BiasAddџ
.transformer_block_9/sequential_9/dense_24/ReluRelu:transformer_block_9/sequential_9/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А20
.transformer_block_9/sequential_9/dense_24/ReluЦ
Btransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype02D
Btransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpЊ
8transformer_block_9/sequential_9/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_9/sequential_9/dense_25/Tensordot/axes≈
8transformer_block_9/sequential_9/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_9/sequential_9/dense_25/Tensordot/freeв
9transformer_block_9/sequential_9/dense_25/Tensordot/ShapeShape<transformer_block_9/sequential_9/dense_24/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_25/Tensordot/Shape»
Atransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2/axis£
<transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_25/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_25/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2ћ
Ctransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1/axis©
>transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_25/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_25/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1ј
9transformer_block_9/sequential_9/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_9/sequential_9/dense_25/Tensordot/Const®
8transformer_block_9/sequential_9/dense_25/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_9/sequential_9/dense_25/Tensordot/Prodƒ
;transformer_block_9/sequential_9/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_9/sequential_9/dense_25/Tensordot/Const_1∞
:transformer_block_9/sequential_9/dense_25/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_9/sequential_9/dense_25/Tensordot/Prod_1ƒ
?transformer_block_9/sequential_9/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_9/sequential_9/dense_25/Tensordot/concat/axisВ
:transformer_block_9/sequential_9/dense_25/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_25/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_25/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_9/sequential_9/dense_25/Tensordot/concatі
9transformer_block_9/sequential_9/dense_25/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_25/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_25/Tensordot/stack≈
=transformer_block_9/sequential_9/dense_25/Tensordot/transpose	Transpose<transformer_block_9/sequential_9/dense_24/Relu:activations:0Ctransformer_block_9/sequential_9/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2?
=transformer_block_9/sequential_9/dense_25/Tensordot/transpose«
;transformer_block_9/sequential_9/dense_25/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_25/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2=
;transformer_block_9/sequential_9/dense_25/Tensordot/Reshape«
:transformer_block_9/sequential_9/dense_25/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_25/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/sequential_9/dense_25/Tensordot/MatMul≈
;transformer_block_9/sequential_9/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2=
;transformer_block_9/sequential_9/dense_25/Tensordot/Const_2»
Atransformer_block_9/sequential_9/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_25/Tensordot/concat_1/axisП
<transformer_block_9/sequential_9/dense_25/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_25/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_25/Tensordot/concat_1є
3transformer_block_9/sequential_9/dense_25/TensordotReshapeDtransformer_block_9/sequential_9/dense_25/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А25
3transformer_block_9/sequential_9/dense_25/TensordotЛ
@transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02B
@transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp∞
1transformer_block_9/sequential_9/dense_25/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_25/Tensordot:output:0Htransformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А23
1transformer_block_9/sequential_9/dense_25/BiasAdd—
'transformer_block_9/dropout_23/IdentityIdentity:transformer_block_9/sequential_9/dense_25/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2)
'transformer_block_9/dropout_23/Identityи
transformer_block_9/add_1AddV2>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:00transformer_block_9/dropout_23/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
transformer_block_9/add_1а
Itransformer_block_9/layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_9/layer_normalization_19/moments/mean/reduction_indicesі
7transformer_block_9/layer_normalization_19/moments/meanMeantransformer_block_9/add_1:z:0Rtransformer_block_9/layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(29
7transformer_block_9/layer_normalization_19/moments/meanК
?transformer_block_9/layer_normalization_19/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2A
?transformer_block_9/layer_normalization_19/moments/StopGradientЅ
Dtransformer_block_9/layer_normalization_19/moments/SquaredDifferenceSquaredDifferencetransformer_block_9/add_1:z:0Htransformer_block_9/layer_normalization_19/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2F
Dtransformer_block_9/layer_normalization_19/moments/SquaredDifferenceи
Mtransformer_block_9/layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_9/layer_normalization_19/moments/variance/reduction_indicesл
;transformer_block_9/layer_normalization_19/moments/varianceMeanHtransformer_block_9/layer_normalization_19/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2=
;transformer_block_9/layer_normalization_19/moments/varianceљ
:transformer_block_9/layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52<
:transformer_block_9/layer_normalization_19/batchnorm/add/yЊ
8transformer_block_9/layer_normalization_19/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_19/moments/variance:output:0Ctransformer_block_9/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2:
8transformer_block_9/layer_normalization_19/batchnorm/addх
:transformer_block_9/layer_normalization_19/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2<
:transformer_block_9/layer_normalization_19/batchnorm/Rsqrt†
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02I
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp√
8transformer_block_9/layer_normalization_19/batchnorm/mulMul>transformer_block_9/layer_normalization_19/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2:
8transformer_block_9/layer_normalization_19/batchnorm/mulУ
:transformer_block_9/layer_normalization_19/batchnorm/mul_1Multransformer_block_9/add_1:z:0<transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_19/batchnorm/mul_1ґ
:transformer_block_9/layer_normalization_19/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_19/moments/mean:output:0<transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_19/batchnorm/mul_2Ф
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02E
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpњ
8transformer_block_9/layer_normalization_19/batchnorm/subSubKtransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_19/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2:
8transformer_block_9/layer_normalization_19/batchnorm/subґ
:transformer_block_9/layer_normalization_19/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_19/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_19/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_19/batchnorm/add_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_2/ConstЊ
flatten_2/ReshapeReshape>transformer_block_9/layer_normalization_19/batchnorm/add_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_2/Reshapex
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisЊ
concatenate_2/concatConcatV2flatten_2/Reshape:output:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€И2
concatenate_2/concat©
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	И@*
dtype02 
dense_26/MatMul/ReadVariableOp•
dense_26/MatMulMatMulconcatenate_2/concat:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/MatMulІ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_26/BiasAdd/ReadVariableOp•
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/ReluЕ
dropout_24/IdentityIdentitydense_26/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_24/Identity®
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_27/MatMul/ReadVariableOp§
dense_27/MatMulMatMuldropout_24/Identity:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_27/MatMulІ
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_27/BiasAdd/ReadVariableOp•
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_27/ReluЕ
dropout_25/IdentityIdentitydense_27/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_25/Identity®
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_28/MatMul/ReadVariableOp§
dense_28/MatMulMatMuldropout_25/Identity:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_28/MatMulІ
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_28/BiasAdd/ReadVariableOp•
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_28/BiasAddШ
IdentityIdentitydense_28/BiasAdd:output:0/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp<^token_and_position_embedding_4/embedding_8/embedding_lookup<^token_and_position_embedding_4/embedding_9/embedding_lookupD^transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpD^transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpO^transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpY^transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpL^transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpD^transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpN^transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpD^transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpN^transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpA^transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpA^transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2z
;token_and_position_embedding_4/embedding_8/embedding_lookup;token_and_position_embedding_4/embedding_8/embedding_lookup2z
;token_and_position_embedding_4/embedding_9/embedding_lookup;token_and_position_embedding_4/embedding_9/embedding_lookup2К
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp2Т
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp2К
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp2Т
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp2†
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpNtransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp2і
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2Ж
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpAtransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp2Ъ
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpKtransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpCtransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp2Ю
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpMtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpCtransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp2Ю
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpMtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2Д
@transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp2И
Btransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOp2Д
@transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp2И
Btransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOp:R N
(
_output_shapes
:€€€€€€€€€ДR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
–
®
5__inference_batch_normalization_4_layer_call_fn_44855

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_430902
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:€€€€€€€€€ДRА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):€€€€€€€€€ДRА::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ДRА
 
_user_specified_nameinputs
Й
О
>__inference_token_and_position_embedding_4_layer_call_fn_44691
x
unknown
	unknown_0
identityИҐStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_430192
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:€€€€€€€€€ДRА2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€ДR::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:€€€€€€€€€ДR

_user_specified_namex
цё
б
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_45130

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
7sequential_9_dense_24_tensordot_readvariableop_resource9
5sequential_9_dense_24_biasadd_readvariableop_resource;
7sequential_9_dense_25_tensordot_readvariableop_resource9
5sequential_9_dense_25_biasadd_readvariableop_resource@
<layer_normalization_19_batchnorm_mul_readvariableop_resource<
8layer_normalization_19_batchnorm_readvariableop_resource
identityИҐ/layer_normalization_18/batchnorm/ReadVariableOpҐ3layer_normalization_18/batchnorm/mul/ReadVariableOpҐ/layer_normalization_19/batchnorm/ReadVariableOpҐ3layer_normalization_19/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_9/attention_output/add/ReadVariableOpҐDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_9/key/add/ReadVariableOpҐ7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_9/query/add/ReadVariableOpҐ9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_9/value/add/ReadVariableOpҐ9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpҐ,sequential_9/dense_24/BiasAdd/ReadVariableOpҐ.sequential_9/dense_24/Tensordot/ReadVariableOpҐ,sequential_9/dense_25/BiasAdd/ReadVariableOpҐ.sequential_9/dense_25/Tensordot/ReadVariableOp€
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02;
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpО
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2,
*multi_head_attention_9/query/einsum/Einsum№
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes
:	А*
dtype021
/multi_head_attention_9/query/add/ReadVariableOpц
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 multi_head_attention_9/query/addщ
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype029
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpИ
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2*
(multi_head_attention_9/key/einsum/Einsum÷
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes
:	А*
dtype02/
-multi_head_attention_9/key/add/ReadVariableOpо
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
multi_head_attention_9/key/add€
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02;
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpО
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2,
*multi_head_attention_9/value/einsum/Einsum№
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes
:	А*
dtype021
/multi_head_attention_9/value/add/ReadVariableOpц
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 multi_head_attention_9/value/addБ
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=2
multi_head_attention_9/Mul/y«
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
multi_head_attention_9/Mulь
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€*
equationaecd,abcd->acbe2&
$multi_head_attention_9/einsum/Einsumƒ
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€2(
&multi_head_attention_9/softmax/Softmax 
'multi_head_attention_9/dropout/IdentityIdentity0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€2)
'multi_head_attention_9/dropout/IdentityХ
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/Identity:output:0$multi_head_attention_9/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationacbe,aecd->abcd2(
&multi_head_attention_9/einsum_1/Einsum†
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02F
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp‘
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€А*
equationabcd,cde->abe27
5multi_head_attention_9/attention_output/einsum/Einsumщ
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:multi_head_attention_9/attention_output/add/ReadVariableOpЮ
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2-
+multi_head_attention_9/attention_output/addЮ
dropout_22/IdentityIdentity/multi_head_attention_9/attention_output/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_22/Identityp
addAddV2inputsdropout_22/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
addЄ
5layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_18/moments/mean/reduction_indicesв
#layer_normalization_18/moments/meanMeanadd:z:0>layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2%
#layer_normalization_18/moments/meanќ
+layer_normalization_18/moments/StopGradientStopGradient,layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2-
+layer_normalization_18/moments/StopGradientп
0layer_normalization_18/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_18/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А22
0layer_normalization_18/moments/SquaredDifferenceј
9layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_18/moments/variance/reduction_indicesЫ
'layer_normalization_18/moments/varianceMean4layer_normalization_18/moments/SquaredDifference:z:0Blayer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2)
'layer_normalization_18/moments/varianceХ
&layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_18/batchnorm/add/yо
$layer_normalization_18/batchnorm/addAddV20layer_normalization_18/moments/variance:output:0/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2&
$layer_normalization_18/batchnorm/addє
&layer_normalization_18/batchnorm/RsqrtRsqrt(layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2(
&layer_normalization_18/batchnorm/Rsqrtд
3layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3layer_normalization_18/batchnorm/mul/ReadVariableOpу
$layer_normalization_18/batchnorm/mulMul*layer_normalization_18/batchnorm/Rsqrt:y:0;layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_18/batchnorm/mulЅ
&layer_normalization_18/batchnorm/mul_1Muladd:z:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/mul_1ж
&layer_normalization_18/batchnorm/mul_2Mul,layer_normalization_18/moments/mean:output:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/mul_2Ў
/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/layer_normalization_18/batchnorm/ReadVariableOpп
$layer_normalization_18/batchnorm/subSub7layer_normalization_18/batchnorm/ReadVariableOp:value:0*layer_normalization_18/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_18/batchnorm/subж
&layer_normalization_18/batchnorm/add_1AddV2*layer_normalization_18/batchnorm/mul_1:z:0(layer_normalization_18/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/add_1Џ
.sequential_9/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_24_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype020
.sequential_9/dense_24/Tensordot/ReadVariableOpЦ
$sequential_9/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_24/Tensordot/axesЭ
$sequential_9/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_24/Tensordot/free®
%sequential_9/dense_24/Tensordot/ShapeShape*layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_9/dense_24/Tensordot/Shape†
-sequential_9/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_24/Tensordot/GatherV2/axisњ
(sequential_9/dense_24/Tensordot/GatherV2GatherV2.sequential_9/dense_24/Tensordot/Shape:output:0-sequential_9/dense_24/Tensordot/free:output:06sequential_9/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_24/Tensordot/GatherV2§
/sequential_9/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_24/Tensordot/GatherV2_1/axis≈
*sequential_9/dense_24/Tensordot/GatherV2_1GatherV2.sequential_9/dense_24/Tensordot/Shape:output:0-sequential_9/dense_24/Tensordot/axes:output:08sequential_9/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_24/Tensordot/GatherV2_1Ш
%sequential_9/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_24/Tensordot/ConstЎ
$sequential_9/dense_24/Tensordot/ProdProd1sequential_9/dense_24/Tensordot/GatherV2:output:0.sequential_9/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_24/Tensordot/ProdЬ
'sequential_9/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_24/Tensordot/Const_1а
&sequential_9/dense_24/Tensordot/Prod_1Prod3sequential_9/dense_24/Tensordot/GatherV2_1:output:00sequential_9/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_24/Tensordot/Prod_1Ь
+sequential_9/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_24/Tensordot/concat/axisЮ
&sequential_9/dense_24/Tensordot/concatConcatV2-sequential_9/dense_24/Tensordot/free:output:0-sequential_9/dense_24/Tensordot/axes:output:04sequential_9/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_24/Tensordot/concatд
%sequential_9/dense_24/Tensordot/stackPack-sequential_9/dense_24/Tensordot/Prod:output:0/sequential_9/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_24/Tensordot/stackч
)sequential_9/dense_24/Tensordot/transpose	Transpose*layer_normalization_18/batchnorm/add_1:z:0/sequential_9/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2+
)sequential_9/dense_24/Tensordot/transposeч
'sequential_9/dense_24/Tensordot/ReshapeReshape-sequential_9/dense_24/Tensordot/transpose:y:0.sequential_9/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_9/dense_24/Tensordot/Reshapeч
&sequential_9/dense_24/Tensordot/MatMulMatMul0sequential_9/dense_24/Tensordot/Reshape:output:06sequential_9/dense_24/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_9/dense_24/Tensordot/MatMulЭ
'sequential_9/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2)
'sequential_9/dense_24/Tensordot/Const_2†
-sequential_9/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_24/Tensordot/concat_1/axisЂ
(sequential_9/dense_24/Tensordot/concat_1ConcatV21sequential_9/dense_24/Tensordot/GatherV2:output:00sequential_9/dense_24/Tensordot/Const_2:output:06sequential_9/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_24/Tensordot/concat_1й
sequential_9/dense_24/TensordotReshape0sequential_9/dense_24/Tensordot/MatMul:product:01sequential_9/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
sequential_9/dense_24/Tensordotѕ
,sequential_9/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_24/BiasAdd/ReadVariableOpа
sequential_9/dense_24/BiasAddBiasAdd(sequential_9/dense_24/Tensordot:output:04sequential_9/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_24/BiasAddЯ
sequential_9/dense_24/ReluRelu&sequential_9/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_24/ReluЏ
.sequential_9/dense_25/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype020
.sequential_9/dense_25/Tensordot/ReadVariableOpЦ
$sequential_9/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_25/Tensordot/axesЭ
$sequential_9/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_25/Tensordot/free¶
%sequential_9/dense_25/Tensordot/ShapeShape(sequential_9/dense_24/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dense_25/Tensordot/Shape†
-sequential_9/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_25/Tensordot/GatherV2/axisњ
(sequential_9/dense_25/Tensordot/GatherV2GatherV2.sequential_9/dense_25/Tensordot/Shape:output:0-sequential_9/dense_25/Tensordot/free:output:06sequential_9/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_25/Tensordot/GatherV2§
/sequential_9/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_25/Tensordot/GatherV2_1/axis≈
*sequential_9/dense_25/Tensordot/GatherV2_1GatherV2.sequential_9/dense_25/Tensordot/Shape:output:0-sequential_9/dense_25/Tensordot/axes:output:08sequential_9/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_25/Tensordot/GatherV2_1Ш
%sequential_9/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_25/Tensordot/ConstЎ
$sequential_9/dense_25/Tensordot/ProdProd1sequential_9/dense_25/Tensordot/GatherV2:output:0.sequential_9/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_25/Tensordot/ProdЬ
'sequential_9/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_25/Tensordot/Const_1а
&sequential_9/dense_25/Tensordot/Prod_1Prod3sequential_9/dense_25/Tensordot/GatherV2_1:output:00sequential_9/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_25/Tensordot/Prod_1Ь
+sequential_9/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_25/Tensordot/concat/axisЮ
&sequential_9/dense_25/Tensordot/concatConcatV2-sequential_9/dense_25/Tensordot/free:output:0-sequential_9/dense_25/Tensordot/axes:output:04sequential_9/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_25/Tensordot/concatд
%sequential_9/dense_25/Tensordot/stackPack-sequential_9/dense_25/Tensordot/Prod:output:0/sequential_9/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_25/Tensordot/stackх
)sequential_9/dense_25/Tensordot/transpose	Transpose(sequential_9/dense_24/Relu:activations:0/sequential_9/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2+
)sequential_9/dense_25/Tensordot/transposeч
'sequential_9/dense_25/Tensordot/ReshapeReshape-sequential_9/dense_25/Tensordot/transpose:y:0.sequential_9/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_9/dense_25/Tensordot/Reshapeч
&sequential_9/dense_25/Tensordot/MatMulMatMul0sequential_9/dense_25/Tensordot/Reshape:output:06sequential_9/dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_9/dense_25/Tensordot/MatMulЭ
'sequential_9/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2)
'sequential_9/dense_25/Tensordot/Const_2†
-sequential_9/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_25/Tensordot/concat_1/axisЂ
(sequential_9/dense_25/Tensordot/concat_1ConcatV21sequential_9/dense_25/Tensordot/GatherV2:output:00sequential_9/dense_25/Tensordot/Const_2:output:06sequential_9/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_25/Tensordot/concat_1й
sequential_9/dense_25/TensordotReshape0sequential_9/dense_25/Tensordot/MatMul:product:01sequential_9/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
sequential_9/dense_25/Tensordotѕ
,sequential_9/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_25/BiasAdd/ReadVariableOpа
sequential_9/dense_25/BiasAddBiasAdd(sequential_9/dense_25/Tensordot:output:04sequential_9/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_25/BiasAddХ
dropout_23/IdentityIdentity&sequential_9/dense_25/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_23/IdentityШ
add_1AddV2*layer_normalization_18/batchnorm/add_1:z:0dropout_23/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
add_1Є
5layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_19/moments/mean/reduction_indicesд
#layer_normalization_19/moments/meanMean	add_1:z:0>layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2%
#layer_normalization_19/moments/meanќ
+layer_normalization_19/moments/StopGradientStopGradient,layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2-
+layer_normalization_19/moments/StopGradientс
0layer_normalization_19/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_19/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А22
0layer_normalization_19/moments/SquaredDifferenceј
9layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_19/moments/variance/reduction_indicesЫ
'layer_normalization_19/moments/varianceMean4layer_normalization_19/moments/SquaredDifference:z:0Blayer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2)
'layer_normalization_19/moments/varianceХ
&layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_19/batchnorm/add/yо
$layer_normalization_19/batchnorm/addAddV20layer_normalization_19/moments/variance:output:0/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2&
$layer_normalization_19/batchnorm/addє
&layer_normalization_19/batchnorm/RsqrtRsqrt(layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2(
&layer_normalization_19/batchnorm/Rsqrtд
3layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3layer_normalization_19/batchnorm/mul/ReadVariableOpу
$layer_normalization_19/batchnorm/mulMul*layer_normalization_19/batchnorm/Rsqrt:y:0;layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_19/batchnorm/mul√
&layer_normalization_19/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/mul_1ж
&layer_normalization_19/batchnorm/mul_2Mul,layer_normalization_19/moments/mean:output:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/mul_2Ў
/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/layer_normalization_19/batchnorm/ReadVariableOpп
$layer_normalization_19/batchnorm/subSub7layer_normalization_19/batchnorm/ReadVariableOp:value:0*layer_normalization_19/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_19/batchnorm/subж
&layer_normalization_19/batchnorm/add_1AddV2*layer_normalization_19/batchnorm/mul_1:z:0(layer_normalization_19/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/add_1Ё
IdentityIdentity*layer_normalization_19/batchnorm/add_1:z:00^layer_normalization_18/batchnorm/ReadVariableOp4^layer_normalization_18/batchnorm/mul/ReadVariableOp0^layer_normalization_19/batchnorm/ReadVariableOp4^layer_normalization_19/batchnorm/mul/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp-^sequential_9/dense_24/BiasAdd/ReadVariableOp/^sequential_9/dense_24/Tensordot/ReadVariableOp-^sequential_9/dense_25/BiasAdd/ReadVariableOp/^sequential_9/dense_25/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€А::::::::::::::::2b
/layer_normalization_18/batchnorm/ReadVariableOp/layer_normalization_18/batchnorm/ReadVariableOp2j
3layer_normalization_18/batchnorm/mul/ReadVariableOp3layer_normalization_18/batchnorm/mul/ReadVariableOp2b
/layer_normalization_19/batchnorm/ReadVariableOp/layer_normalization_19/batchnorm/ReadVariableOp2j
3layer_normalization_19/batchnorm/mul/ReadVariableOp3layer_normalization_19/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2\
,sequential_9/dense_24/BiasAdd/ReadVariableOp,sequential_9/dense_24/BiasAdd/ReadVariableOp2`
.sequential_9/dense_24/Tensordot/ReadVariableOp.sequential_9/dense_24/Tensordot/ReadVariableOp2\
,sequential_9/dense_25/BiasAdd/ReadVariableOp,sequential_9/dense_25/BiasAdd/ReadVariableOp2`
.sequential_9/dense_25/Tensordot/ReadVariableOp.sequential_9/dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
∆
£
'__inference_model_2_layer_call_fn_44596
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

unknown_26
identityИҐStatefulPartitionedCallг
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
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_438302
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:€€€€€€€€€ДR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
Н
d
E__inference_dropout_24_layer_call_and_return_conditional_losses_43580

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeј
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
“
І
,__inference_sequential_9_layer_call_fn_42990
dense_24_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCalldense_24_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_429792
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:€€€€€€€€€А
(
_user_specified_namedense_24_input
ь0
≈
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44727

inputs
assignmovingavg_44702
assignmovingavg_1_44708)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient≤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЈ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/44702*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_44702*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/44702*
_output_shapes	
:А2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/44702*
_output_shapes	
:А2
AssignMovingAvg/mul≠
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_44702AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/44702*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp—
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/44708*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_44708*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/44708*
_output_shapes	
:А2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/44708*
_output_shapes	
:А2
AssignMovingAvg_1/mulє
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_44708AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/44708*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
batchnorm/add_1Ѕ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:€€€€€€€€€€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
¶
Y
-__inference_concatenate_2_layer_call_fn_45228
inputs_0
inputs_1
identity„
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_435322
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€И2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€А:€€€€€€€€€:R N
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
ќ
®
5__inference_batch_normalization_4_layer_call_fn_44842

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_430702
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:€€€€€€€€€ДRА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):€€€€€€€€€ДRА::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ДRА
 
_user_specified_nameinputs
£
З
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_42797

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
batchnorm/add_1й
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:€€€€€€€€€€€€€€€€€€А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
у
Б
Y__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_44682
x&
"embedding_9_embedding_lookup_44669&
"embedding_8_embedding_lookup_44675
identityИҐembedding_8/embedding_lookupҐembedding_9/embedding_lookup?
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
€€€€€€€€€2
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
strided_slice/stack_2в
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
range/deltaА
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:€€€€€€€€€2
rangeЃ
embedding_9/embedding_lookupResourceGather"embedding_9_embedding_lookup_44669range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_9/embedding_lookup/44669*(
_output_shapes
:€€€€€€€€€А*
dtype02
embedding_9/embedding_lookupЩ
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_9/embedding_lookup/44669*(
_output_shapes
:€€€€€€€€€А2'
%embedding_9/embedding_lookup/IdentityЅ
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2)
'embedding_9/embedding_lookup/Identity_1q
embedding_8/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€ДR2
embedding_8/Castє
embedding_8/embedding_lookupResourceGather"embedding_8_embedding_lookup_44675embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_8/embedding_lookup/44675*-
_output_shapes
:€€€€€€€€€ДRА*
dtype02
embedding_8/embedding_lookupЮ
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_8/embedding_lookup/44675*-
_output_shapes
:€€€€€€€€€ДRА2'
%embedding_8/embedding_lookup/Identity∆
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2)
'embedding_8/embedding_lookup/Identity_1ѓ
addAddV20embedding_8/embedding_lookup/Identity_1:output:00embedding_9/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
addЯ
IdentityIdentityadd:z:0^embedding_8/embedding_lookup^embedding_9/embedding_lookup*
T0*-
_output_shapes
:€€€€€€€€€ДRА2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€ДR::2<
embedding_8/embedding_lookupembedding_8/embedding_lookup2<
embedding_9/embedding_lookupembedding_9/embedding_lookup:K G
(
_output_shapes
:€€€€€€€€€ДR

_user_specified_namex
„Д
д:
!__inference__traced_restore_46116
file_prefix0
,assignvariableop_batch_normalization_4_gamma1
-assignvariableop_1_batch_normalization_4_beta8
4assignvariableop_2_batch_normalization_4_moving_mean<
8assignvariableop_3_batch_normalization_4_moving_variance&
"assignvariableop_4_dense_26_kernel$
 assignvariableop_5_dense_26_bias&
"assignvariableop_6_dense_27_kernel$
 assignvariableop_7_dense_27_bias&
"assignvariableop_8_dense_28_kernel$
 assignvariableop_9_dense_28_bias
assignvariableop_10_beta_1
assignvariableop_11_beta_2
assignvariableop_12_decay%
!assignvariableop_13_learning_rate!
assignvariableop_14_adam_iterM
Iassignvariableop_15_token_and_position_embedding_4_embedding_8_embeddingsM
Iassignvariableop_16_token_and_position_embedding_4_embedding_9_embeddingsO
Kassignvariableop_17_transformer_block_9_multi_head_attention_9_query_kernelM
Iassignvariableop_18_transformer_block_9_multi_head_attention_9_query_biasM
Iassignvariableop_19_transformer_block_9_multi_head_attention_9_key_kernelK
Gassignvariableop_20_transformer_block_9_multi_head_attention_9_key_biasO
Kassignvariableop_21_transformer_block_9_multi_head_attention_9_value_kernelM
Iassignvariableop_22_transformer_block_9_multi_head_attention_9_value_biasZ
Vassignvariableop_23_transformer_block_9_multi_head_attention_9_attention_output_kernelX
Tassignvariableop_24_transformer_block_9_multi_head_attention_9_attention_output_bias'
#assignvariableop_25_dense_24_kernel%
!assignvariableop_26_dense_24_bias'
#assignvariableop_27_dense_25_kernel%
!assignvariableop_28_dense_25_biasH
Dassignvariableop_29_transformer_block_9_layer_normalization_18_gammaG
Cassignvariableop_30_transformer_block_9_layer_normalization_18_betaH
Dassignvariableop_31_transformer_block_9_layer_normalization_19_gammaG
Cassignvariableop_32_transformer_block_9_layer_normalization_19_beta
assignvariableop_33_total
assignvariableop_34_count:
6assignvariableop_35_adam_batch_normalization_4_gamma_m9
5assignvariableop_36_adam_batch_normalization_4_beta_m.
*assignvariableop_37_adam_dense_26_kernel_m,
(assignvariableop_38_adam_dense_26_bias_m.
*assignvariableop_39_adam_dense_27_kernel_m,
(assignvariableop_40_adam_dense_27_bias_m.
*assignvariableop_41_adam_dense_28_kernel_m,
(assignvariableop_42_adam_dense_28_bias_mT
Passignvariableop_43_adam_token_and_position_embedding_4_embedding_8_embeddings_mT
Passignvariableop_44_adam_token_and_position_embedding_4_embedding_9_embeddings_mV
Rassignvariableop_45_adam_transformer_block_9_multi_head_attention_9_query_kernel_mT
Passignvariableop_46_adam_transformer_block_9_multi_head_attention_9_query_bias_mT
Passignvariableop_47_adam_transformer_block_9_multi_head_attention_9_key_kernel_mR
Nassignvariableop_48_adam_transformer_block_9_multi_head_attention_9_key_bias_mV
Rassignvariableop_49_adam_transformer_block_9_multi_head_attention_9_value_kernel_mT
Passignvariableop_50_adam_transformer_block_9_multi_head_attention_9_value_bias_ma
]assignvariableop_51_adam_transformer_block_9_multi_head_attention_9_attention_output_kernel_m_
[assignvariableop_52_adam_transformer_block_9_multi_head_attention_9_attention_output_bias_m.
*assignvariableop_53_adam_dense_24_kernel_m,
(assignvariableop_54_adam_dense_24_bias_m.
*assignvariableop_55_adam_dense_25_kernel_m,
(assignvariableop_56_adam_dense_25_bias_mO
Kassignvariableop_57_adam_transformer_block_9_layer_normalization_18_gamma_mN
Jassignvariableop_58_adam_transformer_block_9_layer_normalization_18_beta_mO
Kassignvariableop_59_adam_transformer_block_9_layer_normalization_19_gamma_mN
Jassignvariableop_60_adam_transformer_block_9_layer_normalization_19_beta_m:
6assignvariableop_61_adam_batch_normalization_4_gamma_v9
5assignvariableop_62_adam_batch_normalization_4_beta_v.
*assignvariableop_63_adam_dense_26_kernel_v,
(assignvariableop_64_adam_dense_26_bias_v.
*assignvariableop_65_adam_dense_27_kernel_v,
(assignvariableop_66_adam_dense_27_bias_v.
*assignvariableop_67_adam_dense_28_kernel_v,
(assignvariableop_68_adam_dense_28_bias_vT
Passignvariableop_69_adam_token_and_position_embedding_4_embedding_8_embeddings_vT
Passignvariableop_70_adam_token_and_position_embedding_4_embedding_9_embeddings_vV
Rassignvariableop_71_adam_transformer_block_9_multi_head_attention_9_query_kernel_vT
Passignvariableop_72_adam_transformer_block_9_multi_head_attention_9_query_bias_vT
Passignvariableop_73_adam_transformer_block_9_multi_head_attention_9_key_kernel_vR
Nassignvariableop_74_adam_transformer_block_9_multi_head_attention_9_key_bias_vV
Rassignvariableop_75_adam_transformer_block_9_multi_head_attention_9_value_kernel_vT
Passignvariableop_76_adam_transformer_block_9_multi_head_attention_9_value_bias_va
]assignvariableop_77_adam_transformer_block_9_multi_head_attention_9_attention_output_kernel_v_
[assignvariableop_78_adam_transformer_block_9_multi_head_attention_9_attention_output_bias_v.
*assignvariableop_79_adam_dense_24_kernel_v,
(assignvariableop_80_adam_dense_24_bias_v.
*assignvariableop_81_adam_dense_25_kernel_v,
(assignvariableop_82_adam_dense_25_bias_vO
Kassignvariableop_83_adam_transformer_block_9_layer_normalization_18_gamma_vN
Jassignvariableop_84_adam_transformer_block_9_layer_normalization_18_beta_vO
Kassignvariableop_85_adam_transformer_block_9_layer_normalization_19_gamma_vN
Jassignvariableop_86_adam_transformer_block_9_layer_normalization_19_beta_v
identity_88ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_9х/
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*Б/
valueч.Bф.XB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЅ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*≈
valueїBЄXB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesж
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ц
_output_shapesг
а::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*f
dtypes\
Z2X	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЂ
AssignVariableOpAssignVariableOp,assignvariableop_batch_normalization_4_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1≤
AssignVariableOp_1AssignVariableOp-assignvariableop_1_batch_normalization_4_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2є
AssignVariableOp_2AssignVariableOp4assignvariableop_2_batch_normalization_4_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3љ
AssignVariableOp_3AssignVariableOp8assignvariableop_3_batch_normalization_4_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4І
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_26_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5•
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_26_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6І
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_27_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7•
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_27_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8І
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_28_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9•
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_28_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ґ
AssignVariableOp_10AssignVariableOpassignvariableop_10_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ґ
AssignVariableOp_11AssignVariableOpassignvariableop_11_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12°
AssignVariableOp_12AssignVariableOpassignvariableop_12_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14•
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15—
AssignVariableOp_15AssignVariableOpIassignvariableop_15_token_and_position_embedding_4_embedding_8_embeddingsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16—
AssignVariableOp_16AssignVariableOpIassignvariableop_16_token_and_position_embedding_4_embedding_9_embeddingsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17”
AssignVariableOp_17AssignVariableOpKassignvariableop_17_transformer_block_9_multi_head_attention_9_query_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18—
AssignVariableOp_18AssignVariableOpIassignvariableop_18_transformer_block_9_multi_head_attention_9_query_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19—
AssignVariableOp_19AssignVariableOpIassignvariableop_19_transformer_block_9_multi_head_attention_9_key_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ѕ
AssignVariableOp_20AssignVariableOpGassignvariableop_20_transformer_block_9_multi_head_attention_9_key_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21”
AssignVariableOp_21AssignVariableOpKassignvariableop_21_transformer_block_9_multi_head_attention_9_value_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22—
AssignVariableOp_22AssignVariableOpIassignvariableop_22_transformer_block_9_multi_head_attention_9_value_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ё
AssignVariableOp_23AssignVariableOpVassignvariableop_23_transformer_block_9_multi_head_attention_9_attention_output_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24№
AssignVariableOp_24AssignVariableOpTassignvariableop_24_transformer_block_9_multi_head_attention_9_attention_output_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ђ
AssignVariableOp_25AssignVariableOp#assignvariableop_25_dense_24_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26©
AssignVariableOp_26AssignVariableOp!assignvariableop_26_dense_24_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ђ
AssignVariableOp_27AssignVariableOp#assignvariableop_27_dense_25_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28©
AssignVariableOp_28AssignVariableOp!assignvariableop_28_dense_25_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ћ
AssignVariableOp_29AssignVariableOpDassignvariableop_29_transformer_block_9_layer_normalization_18_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ћ
AssignVariableOp_30AssignVariableOpCassignvariableop_30_transformer_block_9_layer_normalization_18_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ћ
AssignVariableOp_31AssignVariableOpDassignvariableop_31_transformer_block_9_layer_normalization_19_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ћ
AssignVariableOp_32AssignVariableOpCassignvariableop_32_transformer_block_9_layer_normalization_19_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33°
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34°
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Њ
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_4_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36љ
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_batch_normalization_4_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37≤
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_26_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38∞
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_26_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39≤
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_27_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40∞
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_27_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41≤
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_28_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42∞
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_28_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ў
AssignVariableOp_43AssignVariableOpPassignvariableop_43_adam_token_and_position_embedding_4_embedding_8_embeddings_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ў
AssignVariableOp_44AssignVariableOpPassignvariableop_44_adam_token_and_position_embedding_4_embedding_9_embeddings_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Џ
AssignVariableOp_45AssignVariableOpRassignvariableop_45_adam_transformer_block_9_multi_head_attention_9_query_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ў
AssignVariableOp_46AssignVariableOpPassignvariableop_46_adam_transformer_block_9_multi_head_attention_9_query_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ў
AssignVariableOp_47AssignVariableOpPassignvariableop_47_adam_transformer_block_9_multi_head_attention_9_key_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48÷
AssignVariableOp_48AssignVariableOpNassignvariableop_48_adam_transformer_block_9_multi_head_attention_9_key_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Џ
AssignVariableOp_49AssignVariableOpRassignvariableop_49_adam_transformer_block_9_multi_head_attention_9_value_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ў
AssignVariableOp_50AssignVariableOpPassignvariableop_50_adam_transformer_block_9_multi_head_attention_9_value_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51е
AssignVariableOp_51AssignVariableOp]assignvariableop_51_adam_transformer_block_9_multi_head_attention_9_attention_output_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52г
AssignVariableOp_52AssignVariableOp[assignvariableop_52_adam_transformer_block_9_multi_head_attention_9_attention_output_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53≤
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_24_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54∞
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_24_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55≤
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_25_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56∞
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_25_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57”
AssignVariableOp_57AssignVariableOpKassignvariableop_57_adam_transformer_block_9_layer_normalization_18_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58“
AssignVariableOp_58AssignVariableOpJassignvariableop_58_adam_transformer_block_9_layer_normalization_18_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59”
AssignVariableOp_59AssignVariableOpKassignvariableop_59_adam_transformer_block_9_layer_normalization_19_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60“
AssignVariableOp_60AssignVariableOpJassignvariableop_60_adam_transformer_block_9_layer_normalization_19_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Њ
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_4_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62љ
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_4_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63≤
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_26_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64∞
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_26_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65≤
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_27_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66∞
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_27_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67≤
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_28_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68∞
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_28_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Ў
AssignVariableOp_69AssignVariableOpPassignvariableop_69_adam_token_and_position_embedding_4_embedding_8_embeddings_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ў
AssignVariableOp_70AssignVariableOpPassignvariableop_70_adam_token_and_position_embedding_4_embedding_9_embeddings_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Џ
AssignVariableOp_71AssignVariableOpRassignvariableop_71_adam_transformer_block_9_multi_head_attention_9_query_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Ў
AssignVariableOp_72AssignVariableOpPassignvariableop_72_adam_transformer_block_9_multi_head_attention_9_query_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ў
AssignVariableOp_73AssignVariableOpPassignvariableop_73_adam_transformer_block_9_multi_head_attention_9_key_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74÷
AssignVariableOp_74AssignVariableOpNassignvariableop_74_adam_transformer_block_9_multi_head_attention_9_key_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Џ
AssignVariableOp_75AssignVariableOpRassignvariableop_75_adam_transformer_block_9_multi_head_attention_9_value_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76Ў
AssignVariableOp_76AssignVariableOpPassignvariableop_76_adam_transformer_block_9_multi_head_attention_9_value_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77е
AssignVariableOp_77AssignVariableOp]assignvariableop_77_adam_transformer_block_9_multi_head_attention_9_attention_output_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78г
AssignVariableOp_78AssignVariableOp[assignvariableop_78_adam_transformer_block_9_multi_head_attention_9_attention_output_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79≤
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_24_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80∞
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_24_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81≤
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_25_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82∞
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_25_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83”
AssignVariableOp_83AssignVariableOpKassignvariableop_83_adam_transformer_block_9_layer_normalization_18_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84“
AssignVariableOp_84AssignVariableOpJassignvariableop_84_adam_transformer_block_9_layer_normalization_18_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85”
AssignVariableOp_85AssignVariableOpKassignvariableop_85_adam_transformer_block_9_layer_normalization_19_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86“
AssignVariableOp_86AssignVariableOpJassignvariableop_86_adam_transformer_block_9_layer_normalization_19_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_869
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЎ
Identity_87Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_87Ћ
Identity_88IdentityIdentity_87:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_88"#
identity_88Identity_88:output:0*у
_input_shapesб
ё: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ч
F
*__inference_dropout_25_layer_call_fn_45322

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_436422
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ы
Д
G__inference_sequential_9_layer_call_and_return_conditional_losses_42935
dense_24_input
dense_24_42924
dense_24_42926
dense_25_42929
dense_25_42931
identityИҐ dense_24/StatefulPartitionedCallҐ dense_25/StatefulPartitionedCall°
 dense_24/StatefulPartitionedCallStatefulPartitionedCalldense_24_inputdense_24_42924dense_24_42926*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_428582"
 dense_24/StatefulPartitionedCallЉ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_42929dense_25_42931*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_429042"
 dense_25/StatefulPartitionedCall»
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€А::::2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:\ X
,
_output_shapes
:€€€€€€€€€А
(
_user_specified_namedense_24_input
Ї
Я
,__inference_sequential_9_layer_call_fn_45481

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_429792
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ћ0
≈
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44809

inputs
assignmovingavg_44784
assignmovingavg_1_44790)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient™
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesЈ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/44784*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_44784*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/44784*
_output_shapes	
:А2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/44784*
_output_shapes	
:А2
AssignMovingAvg/mul≠
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_44784AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/44784*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp—
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/44790*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_44790*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/44790*
_output_shapes	
:А2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/44790*
_output_shapes	
:А2
AssignMovingAvg_1/mulє
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_44790AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/44790*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mul|
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЛ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2
batchnorm/add_1є
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:€€€€€€€€€ДRА2

Identity"
identityIdentity:output:0*<
_input_shapes+
):€€€€€€€€€ДRА::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€ДRА
 
_user_specified_nameinputs
цё
б
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_43402

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
7sequential_9_dense_24_tensordot_readvariableop_resource9
5sequential_9_dense_24_biasadd_readvariableop_resource;
7sequential_9_dense_25_tensordot_readvariableop_resource9
5sequential_9_dense_25_biasadd_readvariableop_resource@
<layer_normalization_19_batchnorm_mul_readvariableop_resource<
8layer_normalization_19_batchnorm_readvariableop_resource
identityИҐ/layer_normalization_18/batchnorm/ReadVariableOpҐ3layer_normalization_18/batchnorm/mul/ReadVariableOpҐ/layer_normalization_19/batchnorm/ReadVariableOpҐ3layer_normalization_19/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_9/attention_output/add/ReadVariableOpҐDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_9/key/add/ReadVariableOpҐ7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_9/query/add/ReadVariableOpҐ9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_9/value/add/ReadVariableOpҐ9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpҐ,sequential_9/dense_24/BiasAdd/ReadVariableOpҐ.sequential_9/dense_24/Tensordot/ReadVariableOpҐ,sequential_9/dense_25/BiasAdd/ReadVariableOpҐ.sequential_9/dense_25/Tensordot/ReadVariableOp€
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02;
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpО
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2,
*multi_head_attention_9/query/einsum/Einsum№
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes
:	А*
dtype021
/multi_head_attention_9/query/add/ReadVariableOpц
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 multi_head_attention_9/query/addщ
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype029
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpИ
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2*
(multi_head_attention_9/key/einsum/Einsum÷
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes
:	А*
dtype02/
-multi_head_attention_9/key/add/ReadVariableOpо
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
multi_head_attention_9/key/add€
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02;
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpО
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2,
*multi_head_attention_9/value/einsum/Einsum№
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes
:	А*
dtype021
/multi_head_attention_9/value/add/ReadVariableOpц
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 multi_head_attention_9/value/addБ
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=2
multi_head_attention_9/Mul/y«
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
multi_head_attention_9/Mulь
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€*
equationaecd,abcd->acbe2&
$multi_head_attention_9/einsum/Einsumƒ
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€2(
&multi_head_attention_9/softmax/Softmax 
'multi_head_attention_9/dropout/IdentityIdentity0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€2)
'multi_head_attention_9/dropout/IdentityХ
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/Identity:output:0$multi_head_attention_9/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationacbe,aecd->abcd2(
&multi_head_attention_9/einsum_1/Einsum†
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02F
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp‘
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€А*
equationabcd,cde->abe27
5multi_head_attention_9/attention_output/einsum/Einsumщ
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:multi_head_attention_9/attention_output/add/ReadVariableOpЮ
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2-
+multi_head_attention_9/attention_output/addЮ
dropout_22/IdentityIdentity/multi_head_attention_9/attention_output/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_22/Identityp
addAddV2inputsdropout_22/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
addЄ
5layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_18/moments/mean/reduction_indicesв
#layer_normalization_18/moments/meanMeanadd:z:0>layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2%
#layer_normalization_18/moments/meanќ
+layer_normalization_18/moments/StopGradientStopGradient,layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2-
+layer_normalization_18/moments/StopGradientп
0layer_normalization_18/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_18/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А22
0layer_normalization_18/moments/SquaredDifferenceј
9layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_18/moments/variance/reduction_indicesЫ
'layer_normalization_18/moments/varianceMean4layer_normalization_18/moments/SquaredDifference:z:0Blayer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2)
'layer_normalization_18/moments/varianceХ
&layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_18/batchnorm/add/yо
$layer_normalization_18/batchnorm/addAddV20layer_normalization_18/moments/variance:output:0/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2&
$layer_normalization_18/batchnorm/addє
&layer_normalization_18/batchnorm/RsqrtRsqrt(layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2(
&layer_normalization_18/batchnorm/Rsqrtд
3layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3layer_normalization_18/batchnorm/mul/ReadVariableOpу
$layer_normalization_18/batchnorm/mulMul*layer_normalization_18/batchnorm/Rsqrt:y:0;layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_18/batchnorm/mulЅ
&layer_normalization_18/batchnorm/mul_1Muladd:z:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/mul_1ж
&layer_normalization_18/batchnorm/mul_2Mul,layer_normalization_18/moments/mean:output:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/mul_2Ў
/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/layer_normalization_18/batchnorm/ReadVariableOpп
$layer_normalization_18/batchnorm/subSub7layer_normalization_18/batchnorm/ReadVariableOp:value:0*layer_normalization_18/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_18/batchnorm/subж
&layer_normalization_18/batchnorm/add_1AddV2*layer_normalization_18/batchnorm/mul_1:z:0(layer_normalization_18/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/add_1Џ
.sequential_9/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_24_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype020
.sequential_9/dense_24/Tensordot/ReadVariableOpЦ
$sequential_9/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_24/Tensordot/axesЭ
$sequential_9/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_24/Tensordot/free®
%sequential_9/dense_24/Tensordot/ShapeShape*layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_9/dense_24/Tensordot/Shape†
-sequential_9/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_24/Tensordot/GatherV2/axisњ
(sequential_9/dense_24/Tensordot/GatherV2GatherV2.sequential_9/dense_24/Tensordot/Shape:output:0-sequential_9/dense_24/Tensordot/free:output:06sequential_9/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_24/Tensordot/GatherV2§
/sequential_9/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_24/Tensordot/GatherV2_1/axis≈
*sequential_9/dense_24/Tensordot/GatherV2_1GatherV2.sequential_9/dense_24/Tensordot/Shape:output:0-sequential_9/dense_24/Tensordot/axes:output:08sequential_9/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_24/Tensordot/GatherV2_1Ш
%sequential_9/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_24/Tensordot/ConstЎ
$sequential_9/dense_24/Tensordot/ProdProd1sequential_9/dense_24/Tensordot/GatherV2:output:0.sequential_9/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_24/Tensordot/ProdЬ
'sequential_9/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_24/Tensordot/Const_1а
&sequential_9/dense_24/Tensordot/Prod_1Prod3sequential_9/dense_24/Tensordot/GatherV2_1:output:00sequential_9/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_24/Tensordot/Prod_1Ь
+sequential_9/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_24/Tensordot/concat/axisЮ
&sequential_9/dense_24/Tensordot/concatConcatV2-sequential_9/dense_24/Tensordot/free:output:0-sequential_9/dense_24/Tensordot/axes:output:04sequential_9/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_24/Tensordot/concatд
%sequential_9/dense_24/Tensordot/stackPack-sequential_9/dense_24/Tensordot/Prod:output:0/sequential_9/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_24/Tensordot/stackч
)sequential_9/dense_24/Tensordot/transpose	Transpose*layer_normalization_18/batchnorm/add_1:z:0/sequential_9/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2+
)sequential_9/dense_24/Tensordot/transposeч
'sequential_9/dense_24/Tensordot/ReshapeReshape-sequential_9/dense_24/Tensordot/transpose:y:0.sequential_9/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_9/dense_24/Tensordot/Reshapeч
&sequential_9/dense_24/Tensordot/MatMulMatMul0sequential_9/dense_24/Tensordot/Reshape:output:06sequential_9/dense_24/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_9/dense_24/Tensordot/MatMulЭ
'sequential_9/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2)
'sequential_9/dense_24/Tensordot/Const_2†
-sequential_9/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_24/Tensordot/concat_1/axisЂ
(sequential_9/dense_24/Tensordot/concat_1ConcatV21sequential_9/dense_24/Tensordot/GatherV2:output:00sequential_9/dense_24/Tensordot/Const_2:output:06sequential_9/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_24/Tensordot/concat_1й
sequential_9/dense_24/TensordotReshape0sequential_9/dense_24/Tensordot/MatMul:product:01sequential_9/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
sequential_9/dense_24/Tensordotѕ
,sequential_9/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_24/BiasAdd/ReadVariableOpа
sequential_9/dense_24/BiasAddBiasAdd(sequential_9/dense_24/Tensordot:output:04sequential_9/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_24/BiasAddЯ
sequential_9/dense_24/ReluRelu&sequential_9/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_24/ReluЏ
.sequential_9/dense_25/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype020
.sequential_9/dense_25/Tensordot/ReadVariableOpЦ
$sequential_9/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_25/Tensordot/axesЭ
$sequential_9/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_25/Tensordot/free¶
%sequential_9/dense_25/Tensordot/ShapeShape(sequential_9/dense_24/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dense_25/Tensordot/Shape†
-sequential_9/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_25/Tensordot/GatherV2/axisњ
(sequential_9/dense_25/Tensordot/GatherV2GatherV2.sequential_9/dense_25/Tensordot/Shape:output:0-sequential_9/dense_25/Tensordot/free:output:06sequential_9/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_25/Tensordot/GatherV2§
/sequential_9/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_25/Tensordot/GatherV2_1/axis≈
*sequential_9/dense_25/Tensordot/GatherV2_1GatherV2.sequential_9/dense_25/Tensordot/Shape:output:0-sequential_9/dense_25/Tensordot/axes:output:08sequential_9/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_25/Tensordot/GatherV2_1Ш
%sequential_9/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_25/Tensordot/ConstЎ
$sequential_9/dense_25/Tensordot/ProdProd1sequential_9/dense_25/Tensordot/GatherV2:output:0.sequential_9/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_25/Tensordot/ProdЬ
'sequential_9/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_25/Tensordot/Const_1а
&sequential_9/dense_25/Tensordot/Prod_1Prod3sequential_9/dense_25/Tensordot/GatherV2_1:output:00sequential_9/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_25/Tensordot/Prod_1Ь
+sequential_9/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_25/Tensordot/concat/axisЮ
&sequential_9/dense_25/Tensordot/concatConcatV2-sequential_9/dense_25/Tensordot/free:output:0-sequential_9/dense_25/Tensordot/axes:output:04sequential_9/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_25/Tensordot/concatд
%sequential_9/dense_25/Tensordot/stackPack-sequential_9/dense_25/Tensordot/Prod:output:0/sequential_9/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_25/Tensordot/stackх
)sequential_9/dense_25/Tensordot/transpose	Transpose(sequential_9/dense_24/Relu:activations:0/sequential_9/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2+
)sequential_9/dense_25/Tensordot/transposeч
'sequential_9/dense_25/Tensordot/ReshapeReshape-sequential_9/dense_25/Tensordot/transpose:y:0.sequential_9/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_9/dense_25/Tensordot/Reshapeч
&sequential_9/dense_25/Tensordot/MatMulMatMul0sequential_9/dense_25/Tensordot/Reshape:output:06sequential_9/dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_9/dense_25/Tensordot/MatMulЭ
'sequential_9/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2)
'sequential_9/dense_25/Tensordot/Const_2†
-sequential_9/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_25/Tensordot/concat_1/axisЂ
(sequential_9/dense_25/Tensordot/concat_1ConcatV21sequential_9/dense_25/Tensordot/GatherV2:output:00sequential_9/dense_25/Tensordot/Const_2:output:06sequential_9/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_25/Tensordot/concat_1й
sequential_9/dense_25/TensordotReshape0sequential_9/dense_25/Tensordot/MatMul:product:01sequential_9/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
sequential_9/dense_25/Tensordotѕ
,sequential_9/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_25/BiasAdd/ReadVariableOpа
sequential_9/dense_25/BiasAddBiasAdd(sequential_9/dense_25/Tensordot:output:04sequential_9/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_25/BiasAddХ
dropout_23/IdentityIdentity&sequential_9/dense_25/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_23/IdentityШ
add_1AddV2*layer_normalization_18/batchnorm/add_1:z:0dropout_23/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
add_1Є
5layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_19/moments/mean/reduction_indicesд
#layer_normalization_19/moments/meanMean	add_1:z:0>layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2%
#layer_normalization_19/moments/meanќ
+layer_normalization_19/moments/StopGradientStopGradient,layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2-
+layer_normalization_19/moments/StopGradientс
0layer_normalization_19/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_19/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А22
0layer_normalization_19/moments/SquaredDifferenceј
9layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_19/moments/variance/reduction_indicesЫ
'layer_normalization_19/moments/varianceMean4layer_normalization_19/moments/SquaredDifference:z:0Blayer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2)
'layer_normalization_19/moments/varianceХ
&layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_19/batchnorm/add/yо
$layer_normalization_19/batchnorm/addAddV20layer_normalization_19/moments/variance:output:0/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2&
$layer_normalization_19/batchnorm/addє
&layer_normalization_19/batchnorm/RsqrtRsqrt(layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2(
&layer_normalization_19/batchnorm/Rsqrtд
3layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3layer_normalization_19/batchnorm/mul/ReadVariableOpу
$layer_normalization_19/batchnorm/mulMul*layer_normalization_19/batchnorm/Rsqrt:y:0;layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_19/batchnorm/mul√
&layer_normalization_19/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/mul_1ж
&layer_normalization_19/batchnorm/mul_2Mul,layer_normalization_19/moments/mean:output:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/mul_2Ў
/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/layer_normalization_19/batchnorm/ReadVariableOpп
$layer_normalization_19/batchnorm/subSub7layer_normalization_19/batchnorm/ReadVariableOp:value:0*layer_normalization_19/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_19/batchnorm/subж
&layer_normalization_19/batchnorm/add_1AddV2*layer_normalization_19/batchnorm/mul_1:z:0(layer_normalization_19/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/add_1Ё
IdentityIdentity*layer_normalization_19/batchnorm/add_1:z:00^layer_normalization_18/batchnorm/ReadVariableOp4^layer_normalization_18/batchnorm/mul/ReadVariableOp0^layer_normalization_19/batchnorm/ReadVariableOp4^layer_normalization_19/batchnorm/mul/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp-^sequential_9/dense_24/BiasAdd/ReadVariableOp/^sequential_9/dense_24/Tensordot/ReadVariableOp-^sequential_9/dense_25/BiasAdd/ReadVariableOp/^sequential_9/dense_25/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€А::::::::::::::::2b
/layer_normalization_18/batchnorm/ReadVariableOp/layer_normalization_18/batchnorm/ReadVariableOp2j
3layer_normalization_18/batchnorm/mul/ReadVariableOp3layer_normalization_18/batchnorm/mul/ReadVariableOp2b
/layer_normalization_19/batchnorm/ReadVariableOp/layer_normalization_19/batchnorm/ReadVariableOp2j
3layer_normalization_19/batchnorm/mul/ReadVariableOp3layer_normalization_19/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2\
,sequential_9/dense_24/BiasAdd/ReadVariableOp,sequential_9/dense_24/BiasAdd/ReadVariableOp2`
.sequential_9/dense_24/Tensordot/ReadVariableOp.sequential_9/dense_24/Tensordot/ReadVariableOp2\
,sequential_9/dense_25/BiasAdd/ReadVariableOp,sequential_9/dense_25/BiasAdd/ReadVariableOp2`
.sequential_9/dense_25/Tensordot/ReadVariableOp.sequential_9/dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
є
r
H__inference_concatenate_2_layer_call_and_return_conditional_losses_43532

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisА
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€И2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€И2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€А:€€€€€€€€€:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ё
}
(__inference_dense_27_layer_call_fn_45295

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_436092
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
»
c
E__inference_dropout_24_layer_call_and_return_conditional_losses_45265

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
£
З
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44747

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2
batchnorm/add_1й
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:€€€€€€€€€€€€€€€€€€А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
†J
Ѓ
G__inference_sequential_9_layer_call_and_return_conditional_losses_45455

inputs.
*dense_24_tensordot_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource.
*dense_25_tensordot_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource
identityИҐdense_24/BiasAdd/ReadVariableOpҐ!dense_24/Tensordot/ReadVariableOpҐdense_25/BiasAdd/ReadVariableOpҐ!dense_25/Tensordot/ReadVariableOp≥
!dense_24/Tensordot/ReadVariableOpReadVariableOp*dense_24_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!dense_24/Tensordot/ReadVariableOp|
dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_24/Tensordot/axesГ
dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_24/Tensordot/freej
dense_24/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_24/Tensordot/ShapeЖ
 dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/GatherV2/axisю
dense_24/Tensordot/GatherV2GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/free:output:0)dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2К
"dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_24/Tensordot/GatherV2_1/axisД
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
dense_24/Tensordot/Const§
dense_24/Tensordot/ProdProd$dense_24/Tensordot/GatherV2:output:0!dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/ProdВ
dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_1ђ
dense_24/Tensordot/Prod_1Prod&dense_24/Tensordot/GatherV2_1:output:0#dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod_1В
dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_24/Tensordot/concat/axisЁ
dense_24/Tensordot/concatConcatV2 dense_24/Tensordot/free:output:0 dense_24/Tensordot/axes:output:0'dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat∞
dense_24/Tensordot/stackPack dense_24/Tensordot/Prod:output:0"dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/stackђ
dense_24/Tensordot/transpose	Transposeinputs"dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_24/Tensordot/transpose√
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_24/Tensordot/Reshape√
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_24/Tensordot/MatMulГ
dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_24/Tensordot/Const_2Ж
 dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/concat_1/axisк
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat_1µ
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_24/Tensordot®
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_24/BiasAdd/ReadVariableOpђ
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_24/BiasAddx
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_24/Relu≥
!dense_25/Tensordot/ReadVariableOpReadVariableOp*dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype02#
!dense_25/Tensordot/ReadVariableOp|
dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_25/Tensordot/axesГ
dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_25/Tensordot/free
dense_25/Tensordot/ShapeShapedense_24/Relu:activations:0*
T0*
_output_shapes
:2
dense_25/Tensordot/ShapeЖ
 dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/GatherV2/axisю
dense_25/Tensordot/GatherV2GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/free:output:0)dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2К
"dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_25/Tensordot/GatherV2_1/axisД
dense_25/Tensordot/GatherV2_1GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/axes:output:0+dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2_1~
dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const§
dense_25/Tensordot/ProdProd$dense_25/Tensordot/GatherV2:output:0!dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/ProdВ
dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const_1ђ
dense_25/Tensordot/Prod_1Prod&dense_25/Tensordot/GatherV2_1:output:0#dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/Prod_1В
dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_25/Tensordot/concat/axisЁ
dense_25/Tensordot/concatConcatV2 dense_25/Tensordot/free:output:0 dense_25/Tensordot/axes:output:0'dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat∞
dense_25/Tensordot/stackPack dense_25/Tensordot/Prod:output:0"dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/stackЅ
dense_25/Tensordot/transpose	Transposedense_24/Relu:activations:0"dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_25/Tensordot/transpose√
dense_25/Tensordot/ReshapeReshape dense_25/Tensordot/transpose:y:0!dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_25/Tensordot/Reshape√
dense_25/Tensordot/MatMulMatMul#dense_25/Tensordot/Reshape:output:0)dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_25/Tensordot/MatMulГ
dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_25/Tensordot/Const_2Ж
 dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/concat_1/axisк
dense_25/Tensordot/concat_1ConcatV2$dense_25/Tensordot/GatherV2:output:0#dense_25/Tensordot/Const_2:output:0)dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat_1µ
dense_25/TensordotReshape#dense_25/Tensordot/MatMul:product:0$dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_25/Tensordot®
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_25/BiasAdd/ReadVariableOpђ
dense_25/BiasAddBiasAdddense_25/Tensordot:output:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dense_25/BiasAddю
IdentityIdentitydense_25/BiasAdd:output:0 ^dense_24/BiasAdd/ReadVariableOp"^dense_24/Tensordot/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp"^dense_25/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€А::::2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2F
!dense_24/Tensordot/ReadVariableOp!dense_24/Tensordot/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2F
!dense_25/Tensordot/ReadVariableOp!dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Љ 
в
C__inference_dense_24_layer_call_and_return_conditional_losses_45512

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpШ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
АА*
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЯ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
я
}
(__inference_dense_26_layer_call_fn_45248

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_435522
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€И::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€И
 
_user_specified_nameinputs
г
ь
G__inference_sequential_9_layer_call_and_return_conditional_losses_42952

inputs
dense_24_42941
dense_24_42943
dense_25_42946
dense_25_42948
identityИҐ dense_24/StatefulPartitionedCallҐ dense_25/StatefulPartitionedCallЩ
 dense_24/StatefulPartitionedCallStatefulPartitionedCallinputsdense_24_42941dense_24_42943*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_428582"
 dense_24/StatefulPartitionedCallЉ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_42946dense_25_42948*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_429042"
 dense_25/StatefulPartitionedCall»
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€А::::2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Д=
∞	
B__inference_model_2_layer_call_and_return_conditional_losses_43754
input_9
input_10(
$token_and_position_embedding_4_43686(
$token_and_position_embedding_4_43688
batch_normalization_4_43691
batch_normalization_4_43693
batch_normalization_4_43695
batch_normalization_4_43697
transformer_block_9_43701
transformer_block_9_43703
transformer_block_9_43705
transformer_block_9_43707
transformer_block_9_43709
transformer_block_9_43711
transformer_block_9_43713
transformer_block_9_43715
transformer_block_9_43717
transformer_block_9_43719
transformer_block_9_43721
transformer_block_9_43723
transformer_block_9_43725
transformer_block_9_43727
transformer_block_9_43729
transformer_block_9_43731
dense_26_43736
dense_26_43738
dense_27_43742
dense_27_43744
dense_28_43748
dense_28_43750
identityИҐ-batch_normalization_4/StatefulPartitionedCallҐ dense_26/StatefulPartitionedCallҐ dense_27/StatefulPartitionedCallҐ dense_28/StatefulPartitionedCallҐ6token_and_position_embedding_4/StatefulPartitionedCallҐ+transformer_block_9/StatefulPartitionedCallЙ
6token_and_position_embedding_4/StatefulPartitionedCallStatefulPartitionedCallinput_9$token_and_position_embedding_4_43686$token_and_position_embedding_4_43688*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_4301928
6token_and_position_embedding_4/StatefulPartitionedCall“
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0batch_normalization_4_43691batch_normalization_4_43693batch_normalization_4_43695batch_normalization_4_43697*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_430902/
-batch_normalization_4/StatefulPartitionedCallђ
#average_pooling1d_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_428172%
#average_pooling1d_4/PartitionedCallМ
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0transformer_block_9_43701transformer_block_9_43703transformer_block_9_43705transformer_block_9_43707transformer_block_9_43709transformer_block_9_43711transformer_block_9_43713transformer_block_9_43715transformer_block_9_43717transformer_block_9_43719transformer_block_9_43721transformer_block_9_43723transformer_block_9_43725transformer_block_9_43727transformer_block_9_43729transformer_block_9_43731*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_434022-
+transformer_block_9/StatefulPartitionedCallИ
flatten_2/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_435172
flatten_2/PartitionedCallН
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0input_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_435322
concatenate_2/PartitionedCallі
 dense_26/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_26_43736dense_26_43738*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_435522"
 dense_26/StatefulPartitionedCall€
dropout_24/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_435852
dropout_24/PartitionedCall±
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0dense_27_43742dense_27_43744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_436092"
 dense_27/StatefulPartitionedCall€
dropout_25/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_436422
dropout_25/PartitionedCall±
 dense_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_28_43748dense_28_43750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_436652"
 dense_28/StatefulPartitionedCallэ
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall7^token_and_position_embedding_4/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2p
6token_and_position_embedding_4/StatefulPartitionedCall6token_and_position_embedding_4/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_9:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_10
Н
d
E__inference_dropout_25_layer_call_and_return_conditional_losses_45307

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeј
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
с
}
(__inference_dense_25_layer_call_fn_45560

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_429042
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
џ
в
C__inference_dense_25_layer_call_and_return_conditional_losses_45551

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpШ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
АА*
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЯ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
»
£
'__inference_model_2_layer_call_fn_44658
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

unknown_26
identityИҐStatefulPartitionedCallе
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
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_439642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:€€€€€€€€€ДR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
“

я
3__inference_transformer_block_9_layer_call_fn_45204

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
identityИҐStatefulPartitionedCallЅ
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
:€€€€€€€€€А*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_434022
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€А::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
р	
№
C__inference_dense_26_layer_call_and_return_conditional_losses_43552

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	И@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€И::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€И
 
_user_specified_nameinputs
ХА
б
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_45003

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
7sequential_9_dense_24_tensordot_readvariableop_resource9
5sequential_9_dense_24_biasadd_readvariableop_resource;
7sequential_9_dense_25_tensordot_readvariableop_resource9
5sequential_9_dense_25_biasadd_readvariableop_resource@
<layer_normalization_19_batchnorm_mul_readvariableop_resource<
8layer_normalization_19_batchnorm_readvariableop_resource
identityИҐ/layer_normalization_18/batchnorm/ReadVariableOpҐ3layer_normalization_18/batchnorm/mul/ReadVariableOpҐ/layer_normalization_19/batchnorm/ReadVariableOpҐ3layer_normalization_19/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_9/attention_output/add/ReadVariableOpҐDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_9/key/add/ReadVariableOpҐ7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_9/query/add/ReadVariableOpҐ9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_9/value/add/ReadVariableOpҐ9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpҐ,sequential_9/dense_24/BiasAdd/ReadVariableOpҐ.sequential_9/dense_24/Tensordot/ReadVariableOpҐ,sequential_9/dense_25/BiasAdd/ReadVariableOpҐ.sequential_9/dense_25/Tensordot/ReadVariableOp€
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02;
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpО
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2,
*multi_head_attention_9/query/einsum/Einsum№
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes
:	А*
dtype021
/multi_head_attention_9/query/add/ReadVariableOpц
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 multi_head_attention_9/query/addщ
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype029
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpИ
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2*
(multi_head_attention_9/key/einsum/Einsum÷
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes
:	А*
dtype02/
-multi_head_attention_9/key/add/ReadVariableOpо
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
multi_head_attention_9/key/add€
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02;
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpО
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2,
*multi_head_attention_9/value/einsum/Einsum№
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes
:	А*
dtype021
/multi_head_attention_9/value/add/ReadVariableOpц
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 multi_head_attention_9/value/addБ
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=2
multi_head_attention_9/Mul/y«
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
multi_head_attention_9/Mulь
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€*
equationaecd,abcd->acbe2&
$multi_head_attention_9/einsum/Einsumƒ
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€2(
&multi_head_attention_9/softmax/Softmax°
,multi_head_attention_9/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,multi_head_attention_9/dropout/dropout/ConstВ
*multi_head_attention_9/dropout/dropout/MulMul0multi_head_attention_9/softmax/Softmax:softmax:05multi_head_attention_9/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€2,
*multi_head_attention_9/dropout/dropout/MulЉ
,multi_head_attention_9/dropout/dropout/ShapeShape0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_9/dropout/dropout/Shape•
Cmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_9/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
dtype0*

seed2E
Cmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_9/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_9/dropout/dropout/GreaterEqual/y¬
3multi_head_attention_9/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_9/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€25
3multi_head_attention_9/dropout/dropout/GreaterEqualд
+multi_head_attention_9/dropout/dropout/CastCast7multi_head_attention_9/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€2-
+multi_head_attention_9/dropout/dropout/Castю
,multi_head_attention_9/dropout/dropout/Mul_1Mul.multi_head_attention_9/dropout/dropout/Mul:z:0/multi_head_attention_9/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€2.
,multi_head_attention_9/dropout/dropout/Mul_1Х
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/dropout/Mul_1:z:0$multi_head_attention_9/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationacbe,aecd->abcd2(
&multi_head_attention_9/einsum_1/Einsum†
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02F
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp‘
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€А*
equationabcd,cde->abe27
5multi_head_attention_9/attention_output/einsum/Einsumщ
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:multi_head_attention_9/attention_output/add/ReadVariableOpЮ
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2-
+multi_head_attention_9/attention_output/addy
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_22/dropout/Const¬
dropout_22/dropout/MulMul/multi_head_attention_9/attention_output/add:z:0!dropout_22/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_22/dropout/MulУ
dropout_22/dropout/ShapeShape/multi_head_attention_9/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shapeу
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А*
dtype0*

seed*
seed221
/dropout_22/dropout/random_uniform/RandomUniformЛ
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_22/dropout/GreaterEqual/yп
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
dropout_22/dropout/GreaterEqual•
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€А2
dropout_22/dropout/CastЂ
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_22/dropout/Mul_1p
addAddV2inputsdropout_22/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2
addЄ
5layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_18/moments/mean/reduction_indicesв
#layer_normalization_18/moments/meanMeanadd:z:0>layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2%
#layer_normalization_18/moments/meanќ
+layer_normalization_18/moments/StopGradientStopGradient,layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2-
+layer_normalization_18/moments/StopGradientп
0layer_normalization_18/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_18/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А22
0layer_normalization_18/moments/SquaredDifferenceј
9layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_18/moments/variance/reduction_indicesЫ
'layer_normalization_18/moments/varianceMean4layer_normalization_18/moments/SquaredDifference:z:0Blayer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2)
'layer_normalization_18/moments/varianceХ
&layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_18/batchnorm/add/yо
$layer_normalization_18/batchnorm/addAddV20layer_normalization_18/moments/variance:output:0/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2&
$layer_normalization_18/batchnorm/addє
&layer_normalization_18/batchnorm/RsqrtRsqrt(layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2(
&layer_normalization_18/batchnorm/Rsqrtд
3layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3layer_normalization_18/batchnorm/mul/ReadVariableOpу
$layer_normalization_18/batchnorm/mulMul*layer_normalization_18/batchnorm/Rsqrt:y:0;layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_18/batchnorm/mulЅ
&layer_normalization_18/batchnorm/mul_1Muladd:z:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/mul_1ж
&layer_normalization_18/batchnorm/mul_2Mul,layer_normalization_18/moments/mean:output:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/mul_2Ў
/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/layer_normalization_18/batchnorm/ReadVariableOpп
$layer_normalization_18/batchnorm/subSub7layer_normalization_18/batchnorm/ReadVariableOp:value:0*layer_normalization_18/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_18/batchnorm/subж
&layer_normalization_18/batchnorm/add_1AddV2*layer_normalization_18/batchnorm/mul_1:z:0(layer_normalization_18/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/add_1Џ
.sequential_9/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_24_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype020
.sequential_9/dense_24/Tensordot/ReadVariableOpЦ
$sequential_9/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_24/Tensordot/axesЭ
$sequential_9/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_24/Tensordot/free®
%sequential_9/dense_24/Tensordot/ShapeShape*layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_9/dense_24/Tensordot/Shape†
-sequential_9/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_24/Tensordot/GatherV2/axisњ
(sequential_9/dense_24/Tensordot/GatherV2GatherV2.sequential_9/dense_24/Tensordot/Shape:output:0-sequential_9/dense_24/Tensordot/free:output:06sequential_9/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_24/Tensordot/GatherV2§
/sequential_9/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_24/Tensordot/GatherV2_1/axis≈
*sequential_9/dense_24/Tensordot/GatherV2_1GatherV2.sequential_9/dense_24/Tensordot/Shape:output:0-sequential_9/dense_24/Tensordot/axes:output:08sequential_9/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_24/Tensordot/GatherV2_1Ш
%sequential_9/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_24/Tensordot/ConstЎ
$sequential_9/dense_24/Tensordot/ProdProd1sequential_9/dense_24/Tensordot/GatherV2:output:0.sequential_9/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_24/Tensordot/ProdЬ
'sequential_9/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_24/Tensordot/Const_1а
&sequential_9/dense_24/Tensordot/Prod_1Prod3sequential_9/dense_24/Tensordot/GatherV2_1:output:00sequential_9/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_24/Tensordot/Prod_1Ь
+sequential_9/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_24/Tensordot/concat/axisЮ
&sequential_9/dense_24/Tensordot/concatConcatV2-sequential_9/dense_24/Tensordot/free:output:0-sequential_9/dense_24/Tensordot/axes:output:04sequential_9/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_24/Tensordot/concatд
%sequential_9/dense_24/Tensordot/stackPack-sequential_9/dense_24/Tensordot/Prod:output:0/sequential_9/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_24/Tensordot/stackч
)sequential_9/dense_24/Tensordot/transpose	Transpose*layer_normalization_18/batchnorm/add_1:z:0/sequential_9/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2+
)sequential_9/dense_24/Tensordot/transposeч
'sequential_9/dense_24/Tensordot/ReshapeReshape-sequential_9/dense_24/Tensordot/transpose:y:0.sequential_9/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_9/dense_24/Tensordot/Reshapeч
&sequential_9/dense_24/Tensordot/MatMulMatMul0sequential_9/dense_24/Tensordot/Reshape:output:06sequential_9/dense_24/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_9/dense_24/Tensordot/MatMulЭ
'sequential_9/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2)
'sequential_9/dense_24/Tensordot/Const_2†
-sequential_9/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_24/Tensordot/concat_1/axisЂ
(sequential_9/dense_24/Tensordot/concat_1ConcatV21sequential_9/dense_24/Tensordot/GatherV2:output:00sequential_9/dense_24/Tensordot/Const_2:output:06sequential_9/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_24/Tensordot/concat_1й
sequential_9/dense_24/TensordotReshape0sequential_9/dense_24/Tensordot/MatMul:product:01sequential_9/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
sequential_9/dense_24/Tensordotѕ
,sequential_9/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_24/BiasAdd/ReadVariableOpа
sequential_9/dense_24/BiasAddBiasAdd(sequential_9/dense_24/Tensordot:output:04sequential_9/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_24/BiasAddЯ
sequential_9/dense_24/ReluRelu&sequential_9/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_24/ReluЏ
.sequential_9/dense_25/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype020
.sequential_9/dense_25/Tensordot/ReadVariableOpЦ
$sequential_9/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_25/Tensordot/axesЭ
$sequential_9/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_25/Tensordot/free¶
%sequential_9/dense_25/Tensordot/ShapeShape(sequential_9/dense_24/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dense_25/Tensordot/Shape†
-sequential_9/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_25/Tensordot/GatherV2/axisњ
(sequential_9/dense_25/Tensordot/GatherV2GatherV2.sequential_9/dense_25/Tensordot/Shape:output:0-sequential_9/dense_25/Tensordot/free:output:06sequential_9/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_25/Tensordot/GatherV2§
/sequential_9/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_25/Tensordot/GatherV2_1/axis≈
*sequential_9/dense_25/Tensordot/GatherV2_1GatherV2.sequential_9/dense_25/Tensordot/Shape:output:0-sequential_9/dense_25/Tensordot/axes:output:08sequential_9/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_25/Tensordot/GatherV2_1Ш
%sequential_9/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_25/Tensordot/ConstЎ
$sequential_9/dense_25/Tensordot/ProdProd1sequential_9/dense_25/Tensordot/GatherV2:output:0.sequential_9/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_25/Tensordot/ProdЬ
'sequential_9/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_25/Tensordot/Const_1а
&sequential_9/dense_25/Tensordot/Prod_1Prod3sequential_9/dense_25/Tensordot/GatherV2_1:output:00sequential_9/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_25/Tensordot/Prod_1Ь
+sequential_9/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_25/Tensordot/concat/axisЮ
&sequential_9/dense_25/Tensordot/concatConcatV2-sequential_9/dense_25/Tensordot/free:output:0-sequential_9/dense_25/Tensordot/axes:output:04sequential_9/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_25/Tensordot/concatд
%sequential_9/dense_25/Tensordot/stackPack-sequential_9/dense_25/Tensordot/Prod:output:0/sequential_9/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_25/Tensordot/stackх
)sequential_9/dense_25/Tensordot/transpose	Transpose(sequential_9/dense_24/Relu:activations:0/sequential_9/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2+
)sequential_9/dense_25/Tensordot/transposeч
'sequential_9/dense_25/Tensordot/ReshapeReshape-sequential_9/dense_25/Tensordot/transpose:y:0.sequential_9/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_9/dense_25/Tensordot/Reshapeч
&sequential_9/dense_25/Tensordot/MatMulMatMul0sequential_9/dense_25/Tensordot/Reshape:output:06sequential_9/dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_9/dense_25/Tensordot/MatMulЭ
'sequential_9/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2)
'sequential_9/dense_25/Tensordot/Const_2†
-sequential_9/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_25/Tensordot/concat_1/axisЂ
(sequential_9/dense_25/Tensordot/concat_1ConcatV21sequential_9/dense_25/Tensordot/GatherV2:output:00sequential_9/dense_25/Tensordot/Const_2:output:06sequential_9/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_25/Tensordot/concat_1й
sequential_9/dense_25/TensordotReshape0sequential_9/dense_25/Tensordot/MatMul:product:01sequential_9/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
sequential_9/dense_25/Tensordotѕ
,sequential_9/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_25/BiasAdd/ReadVariableOpа
sequential_9/dense_25/BiasAddBiasAdd(sequential_9/dense_25/Tensordot:output:04sequential_9/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_25/BiasAddy
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_23/dropout/Constє
dropout_23/dropout/MulMul&sequential_9/dense_25/BiasAdd:output:0!dropout_23/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_23/dropout/MulК
dropout_23/dropout/ShapeShape&sequential_9/dense_25/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shapeу
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А*
dtype0*

seed*
seed221
/dropout_23/dropout/random_uniform/RandomUniformЛ
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_23/dropout/GreaterEqual/yп
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
dropout_23/dropout/GreaterEqual•
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€А2
dropout_23/dropout/CastЂ
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_23/dropout/Mul_1Ш
add_1AddV2*layer_normalization_18/batchnorm/add_1:z:0dropout_23/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2
add_1Є
5layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_19/moments/mean/reduction_indicesд
#layer_normalization_19/moments/meanMean	add_1:z:0>layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2%
#layer_normalization_19/moments/meanќ
+layer_normalization_19/moments/StopGradientStopGradient,layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2-
+layer_normalization_19/moments/StopGradientс
0layer_normalization_19/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_19/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А22
0layer_normalization_19/moments/SquaredDifferenceј
9layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_19/moments/variance/reduction_indicesЫ
'layer_normalization_19/moments/varianceMean4layer_normalization_19/moments/SquaredDifference:z:0Blayer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2)
'layer_normalization_19/moments/varianceХ
&layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_19/batchnorm/add/yо
$layer_normalization_19/batchnorm/addAddV20layer_normalization_19/moments/variance:output:0/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2&
$layer_normalization_19/batchnorm/addє
&layer_normalization_19/batchnorm/RsqrtRsqrt(layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2(
&layer_normalization_19/batchnorm/Rsqrtд
3layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3layer_normalization_19/batchnorm/mul/ReadVariableOpу
$layer_normalization_19/batchnorm/mulMul*layer_normalization_19/batchnorm/Rsqrt:y:0;layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_19/batchnorm/mul√
&layer_normalization_19/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/mul_1ж
&layer_normalization_19/batchnorm/mul_2Mul,layer_normalization_19/moments/mean:output:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/mul_2Ў
/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/layer_normalization_19/batchnorm/ReadVariableOpп
$layer_normalization_19/batchnorm/subSub7layer_normalization_19/batchnorm/ReadVariableOp:value:0*layer_normalization_19/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_19/batchnorm/subж
&layer_normalization_19/batchnorm/add_1AddV2*layer_normalization_19/batchnorm/mul_1:z:0(layer_normalization_19/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/add_1Ё
IdentityIdentity*layer_normalization_19/batchnorm/add_1:z:00^layer_normalization_18/batchnorm/ReadVariableOp4^layer_normalization_18/batchnorm/mul/ReadVariableOp0^layer_normalization_19/batchnorm/ReadVariableOp4^layer_normalization_19/batchnorm/mul/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp-^sequential_9/dense_24/BiasAdd/ReadVariableOp/^sequential_9/dense_24/Tensordot/ReadVariableOp-^sequential_9/dense_25/BiasAdd/ReadVariableOp/^sequential_9/dense_25/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€А::::::::::::::::2b
/layer_normalization_18/batchnorm/ReadVariableOp/layer_normalization_18/batchnorm/ReadVariableOp2j
3layer_normalization_18/batchnorm/mul/ReadVariableOp3layer_normalization_18/batchnorm/mul/ReadVariableOp2b
/layer_normalization_19/batchnorm/ReadVariableOp/layer_normalization_19/batchnorm/ReadVariableOp2j
3layer_normalization_19/batchnorm/mul/ReadVariableOp3layer_normalization_19/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2\
,sequential_9/dense_24/BiasAdd/ReadVariableOp,sequential_9/dense_24/BiasAdd/ReadVariableOp2`
.sequential_9/dense_24/Tensordot/ReadVariableOp.sequential_9/dense_24/Tensordot/ReadVariableOp2\
,sequential_9/dense_25/BiasAdd/ReadVariableOp,sequential_9/dense_25/BiasAdd/ReadVariableOp2`
.sequential_9/dense_25/Tensordot/ReadVariableOp.sequential_9/dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ы
Д
G__inference_sequential_9_layer_call_and_return_conditional_losses_42921
dense_24_input
dense_24_42869
dense_24_42871
dense_25_42915
dense_25_42917
identityИҐ dense_24/StatefulPartitionedCallҐ dense_25/StatefulPartitionedCall°
 dense_24/StatefulPartitionedCallStatefulPartitionedCalldense_24_inputdense_24_42869dense_24_42871*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_428582"
 dense_24/StatefulPartitionedCallЉ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_42915dense_25_42917*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_429042"
 dense_25/StatefulPartitionedCall»
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€А::::2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:\ X
,
_output_shapes
:€€€€€€€€€А
(
_user_specified_namedense_24_input
Я
Ю
#__inference_signature_wrapper_44095
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

unknown_26
identityИҐStatefulPartitionedCall¬
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
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_426682
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€:€€€€€€€€€ДR::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_10:QM
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_9
с
}
(__inference_dense_24_layer_call_fn_45521

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_428582
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
С	
№
C__inference_dense_28_layer_call_and_return_conditional_losses_45332

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
К@
щ	
B__inference_model_2_layer_call_and_return_conditional_losses_43830

inputs
inputs_1(
$token_and_position_embedding_4_43762(
$token_and_position_embedding_4_43764
batch_normalization_4_43767
batch_normalization_4_43769
batch_normalization_4_43771
batch_normalization_4_43773
transformer_block_9_43777
transformer_block_9_43779
transformer_block_9_43781
transformer_block_9_43783
transformer_block_9_43785
transformer_block_9_43787
transformer_block_9_43789
transformer_block_9_43791
transformer_block_9_43793
transformer_block_9_43795
transformer_block_9_43797
transformer_block_9_43799
transformer_block_9_43801
transformer_block_9_43803
transformer_block_9_43805
transformer_block_9_43807
dense_26_43812
dense_26_43814
dense_27_43818
dense_27_43820
dense_28_43824
dense_28_43826
identityИҐ-batch_normalization_4/StatefulPartitionedCallҐ dense_26/StatefulPartitionedCallҐ dense_27/StatefulPartitionedCallҐ dense_28/StatefulPartitionedCallҐ"dropout_24/StatefulPartitionedCallҐ"dropout_25/StatefulPartitionedCallҐ6token_and_position_embedding_4/StatefulPartitionedCallҐ+transformer_block_9/StatefulPartitionedCallИ
6token_and_position_embedding_4/StatefulPartitionedCallStatefulPartitionedCallinputs$token_and_position_embedding_4_43762$token_and_position_embedding_4_43764*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_4301928
6token_and_position_embedding_4/StatefulPartitionedCall–
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0batch_normalization_4_43767batch_normalization_4_43769batch_normalization_4_43771batch_normalization_4_43773*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_430702/
-batch_normalization_4/StatefulPartitionedCallђ
#average_pooling1d_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_428172%
#average_pooling1d_4/PartitionedCallМ
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0transformer_block_9_43777transformer_block_9_43779transformer_block_9_43781transformer_block_9_43783transformer_block_9_43785transformer_block_9_43787transformer_block_9_43789transformer_block_9_43791transformer_block_9_43793transformer_block_9_43795transformer_block_9_43797transformer_block_9_43799transformer_block_9_43801transformer_block_9_43803transformer_block_9_43805transformer_block_9_43807*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_432752-
+transformer_block_9/StatefulPartitionedCallИ
flatten_2/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_435172
flatten_2/PartitionedCallН
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_435322
concatenate_2/PartitionedCallі
 dense_26/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_26_43812dense_26_43814*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_435522"
 dense_26/StatefulPartitionedCallЧ
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_435802$
"dropout_24/StatefulPartitionedCallє
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0dense_27_43818dense_27_43820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_436092"
 dense_27/StatefulPartitionedCallЉ
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_436372$
"dropout_25/StatefulPartitionedCallє
 dense_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_28_43824dense_28_43826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_436652"
 dense_28/StatefulPartitionedCall«
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall7^token_and_position_embedding_4/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2p
6token_and_position_embedding_4/StatefulPartitionedCall6token_and_position_embedding_4/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ДR
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ц
j
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_42817

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDimsЉ
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
И'*
paddingVALID*
strides	
И'2	
AvgPoolО
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
н	
№
C__inference_dense_27_layer_call_and_return_conditional_losses_43609

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ґ
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_43517

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€А:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
П@
ъ	
B__inference_model_2_layer_call_and_return_conditional_losses_43682
input_9
input_10(
$token_and_position_embedding_4_43030(
$token_and_position_embedding_4_43032
batch_normalization_4_43117
batch_normalization_4_43119
batch_normalization_4_43121
batch_normalization_4_43123
transformer_block_9_43478
transformer_block_9_43480
transformer_block_9_43482
transformer_block_9_43484
transformer_block_9_43486
transformer_block_9_43488
transformer_block_9_43490
transformer_block_9_43492
transformer_block_9_43494
transformer_block_9_43496
transformer_block_9_43498
transformer_block_9_43500
transformer_block_9_43502
transformer_block_9_43504
transformer_block_9_43506
transformer_block_9_43508
dense_26_43563
dense_26_43565
dense_27_43620
dense_27_43622
dense_28_43676
dense_28_43678
identityИҐ-batch_normalization_4/StatefulPartitionedCallҐ dense_26/StatefulPartitionedCallҐ dense_27/StatefulPartitionedCallҐ dense_28/StatefulPartitionedCallҐ"dropout_24/StatefulPartitionedCallҐ"dropout_25/StatefulPartitionedCallҐ6token_and_position_embedding_4/StatefulPartitionedCallҐ+transformer_block_9/StatefulPartitionedCallЙ
6token_and_position_embedding_4/StatefulPartitionedCallStatefulPartitionedCallinput_9$token_and_position_embedding_4_43030$token_and_position_embedding_4_43032*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_4301928
6token_and_position_embedding_4/StatefulPartitionedCall–
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_4/StatefulPartitionedCall:output:0batch_normalization_4_43117batch_normalization_4_43119batch_normalization_4_43121batch_normalization_4_43123*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€ДRА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_430702/
-batch_normalization_4/StatefulPartitionedCallђ
#average_pooling1d_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_428172%
#average_pooling1d_4/PartitionedCallМ
+transformer_block_9/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0transformer_block_9_43478transformer_block_9_43480transformer_block_9_43482transformer_block_9_43484transformer_block_9_43486transformer_block_9_43488transformer_block_9_43490transformer_block_9_43492transformer_block_9_43494transformer_block_9_43496transformer_block_9_43498transformer_block_9_43500transformer_block_9_43502transformer_block_9_43504transformer_block_9_43506transformer_block_9_43508*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_432752-
+transformer_block_9/StatefulPartitionedCallИ
flatten_2/PartitionedCallPartitionedCall4transformer_block_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_435172
flatten_2/PartitionedCallН
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0input_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_435322
concatenate_2/PartitionedCallі
 dense_26/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_26_43563dense_26_43565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_26_layer_call_and_return_conditional_losses_435522"
 dense_26/StatefulPartitionedCallЧ
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_435802$
"dropout_24/StatefulPartitionedCallє
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0dense_27_43620dense_27_43622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_27_layer_call_and_return_conditional_losses_436092"
 dense_27/StatefulPartitionedCallЉ
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_436372$
"dropout_25/StatefulPartitionedCallє
 dense_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_28_43676dense_28_43678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_28_layer_call_and_return_conditional_losses_436652"
 dense_28/StatefulPartitionedCall«
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall7^token_and_position_embedding_4/StatefulPartitionedCall,^transformer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2p
6token_and_position_embedding_4/StatefulPartitionedCall6token_and_position_embedding_4/StatefulPartitionedCall2Z
+transformer_block_9/StatefulPartitionedCall+transformer_block_9/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_9:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_10
Н
d
E__inference_dropout_25_layer_call_and_return_conditional_losses_43637

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeј
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
г
ь
G__inference_sequential_9_layer_call_and_return_conditional_losses_42979

inputs
dense_24_42968
dense_24_42970
dense_25_42973
dense_25_42975
identityИҐ dense_24/StatefulPartitionedCallҐ dense_25/StatefulPartitionedCallЩ
 dense_24/StatefulPartitionedCallStatefulPartitionedCallinputsdense_24_42968dense_24_42970*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_428582"
 dense_24/StatefulPartitionedCallЉ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_42973dense_25_42975*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_25_layer_call_and_return_conditional_losses_429042"
 dense_25/StatefulPartitionedCall»
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€А::::2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
£
c
*__inference_dropout_25_layer_call_fn_45317

inputs
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_436372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Љ 
в
C__inference_dense_24_layer_call_and_return_conditional_losses_42858

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpШ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
АА*
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЯ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ч
F
*__inference_dropout_24_layer_call_fn_45275

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_435852
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
р
®
5__inference_batch_normalization_4_layer_call_fn_44773

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_427972
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
В
O
3__inference_average_pooling1d_4_layer_call_fn_42823

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_428172
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
вЇ
Ѓ!
 __inference__wrapped_model_42668
input_9
input_10M
Imodel_2_token_and_position_embedding_4_embedding_9_embedding_lookup_42486M
Imodel_2_token_and_position_embedding_4_embedding_8_embedding_lookup_42492C
?model_2_batch_normalization_4_batchnorm_readvariableop_resourceG
Cmodel_2_batch_normalization_4_batchnorm_mul_readvariableop_resourceE
Amodel_2_batch_normalization_4_batchnorm_readvariableop_1_resourceE
Amodel_2_batch_normalization_4_batchnorm_readvariableop_2_resourceb
^model_2_transformer_block_9_multi_head_attention_9_query_einsum_einsum_readvariableop_resourceX
Tmodel_2_transformer_block_9_multi_head_attention_9_query_add_readvariableop_resource`
\model_2_transformer_block_9_multi_head_attention_9_key_einsum_einsum_readvariableop_resourceV
Rmodel_2_transformer_block_9_multi_head_attention_9_key_add_readvariableop_resourceb
^model_2_transformer_block_9_multi_head_attention_9_value_einsum_einsum_readvariableop_resourceX
Tmodel_2_transformer_block_9_multi_head_attention_9_value_add_readvariableop_resourcem
imodel_2_transformer_block_9_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resourcec
_model_2_transformer_block_9_multi_head_attention_9_attention_output_add_readvariableop_resource\
Xmodel_2_transformer_block_9_layer_normalization_18_batchnorm_mul_readvariableop_resourceX
Tmodel_2_transformer_block_9_layer_normalization_18_batchnorm_readvariableop_resourceW
Smodel_2_transformer_block_9_sequential_9_dense_24_tensordot_readvariableop_resourceU
Qmodel_2_transformer_block_9_sequential_9_dense_24_biasadd_readvariableop_resourceW
Smodel_2_transformer_block_9_sequential_9_dense_25_tensordot_readvariableop_resourceU
Qmodel_2_transformer_block_9_sequential_9_dense_25_biasadd_readvariableop_resource\
Xmodel_2_transformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resourceX
Tmodel_2_transformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource3
/model_2_dense_26_matmul_readvariableop_resource4
0model_2_dense_26_biasadd_readvariableop_resource3
/model_2_dense_27_matmul_readvariableop_resource4
0model_2_dense_27_biasadd_readvariableop_resource3
/model_2_dense_28_matmul_readvariableop_resource4
0model_2_dense_28_biasadd_readvariableop_resource
identityИҐ6model_2/batch_normalization_4/batchnorm/ReadVariableOpҐ8model_2/batch_normalization_4/batchnorm/ReadVariableOp_1Ґ8model_2/batch_normalization_4/batchnorm/ReadVariableOp_2Ґ:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOpҐ'model_2/dense_26/BiasAdd/ReadVariableOpҐ&model_2/dense_26/MatMul/ReadVariableOpҐ'model_2/dense_27/BiasAdd/ReadVariableOpҐ&model_2/dense_27/MatMul/ReadVariableOpҐ'model_2/dense_28/BiasAdd/ReadVariableOpҐ&model_2/dense_28/MatMul/ReadVariableOpҐCmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookupҐCmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookupҐKmodel_2/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpҐOmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpҐKmodel_2/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpҐOmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpҐVmodel_2/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpҐ`model_2/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpҐImodel_2/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpҐSmodel_2/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpҐKmodel_2/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpҐUmodel_2/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpҐKmodel_2/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpҐUmodel_2/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpҐHmodel_2/transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOpҐJmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpҐHmodel_2/transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOpҐJmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpУ
,model_2/token_and_position_embedding_4/ShapeShapeinput_9*
T0*
_output_shapes
:2.
,model_2/token_and_position_embedding_4/ShapeЋ
:model_2/token_and_position_embedding_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2<
:model_2/token_and_position_embedding_4/strided_slice/stack∆
<model_2/token_and_position_embedding_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_2/token_and_position_embedding_4/strided_slice/stack_1∆
<model_2/token_and_position_embedding_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_2/token_and_position_embedding_4/strided_slice/stack_2ћ
4model_2/token_and_position_embedding_4/strided_sliceStridedSlice5model_2/token_and_position_embedding_4/Shape:output:0Cmodel_2/token_and_position_embedding_4/strided_slice/stack:output:0Emodel_2/token_and_position_embedding_4/strided_slice/stack_1:output:0Emodel_2/token_and_position_embedding_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_2/token_and_position_embedding_4/strided_slice™
2model_2/token_and_position_embedding_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_2/token_and_position_embedding_4/range/start™
2model_2/token_and_position_embedding_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_2/token_and_position_embedding_4/range/delta√
,model_2/token_and_position_embedding_4/rangeRange;model_2/token_and_position_embedding_4/range/start:output:0=model_2/token_and_position_embedding_4/strided_slice:output:0;model_2/token_and_position_embedding_4/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2.
,model_2/token_and_position_embedding_4/rangeс
Cmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookupResourceGatherImodel_2_token_and_position_embedding_4_embedding_9_embedding_lookup_424865model_2/token_and_position_embedding_4/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*\
_classR
PNloc:@model_2/token_and_position_embedding_4/embedding_9/embedding_lookup/42486*(
_output_shapes
:€€€€€€€€€А*
dtype02E
Cmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookupµ
Lmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*\
_classR
PNloc:@model_2/token_and_position_embedding_4/embedding_9/embedding_lookup/42486*(
_output_shapes
:€€€€€€€€€А2N
Lmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookup/Identityґ
Nmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2P
Nmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1≈
7model_2/token_and_position_embedding_4/embedding_8/CastCastinput_9*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€ДR29
7model_2/token_and_position_embedding_4/embedding_8/Castь
Cmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookupResourceGatherImodel_2_token_and_position_embedding_4_embedding_8_embedding_lookup_42492;model_2/token_and_position_embedding_4/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*\
_classR
PNloc:@model_2/token_and_position_embedding_4/embedding_8/embedding_lookup/42492*-
_output_shapes
:€€€€€€€€€ДRА*
dtype02E
Cmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookupЇ
Lmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*\
_classR
PNloc:@model_2/token_and_position_embedding_4/embedding_8/embedding_lookup/42492*-
_output_shapes
:€€€€€€€€€ДRА2N
Lmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookup/Identityї
Nmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2P
Nmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1Ћ
*model_2/token_and_position_embedding_4/addAddV2Wmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1:output:0Wmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2,
*model_2/token_and_position_embedding_4/addн
6model_2/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype028
6model_2/batch_normalization_4/batchnorm/ReadVariableOp£
-model_2/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-model_2/batch_normalization_4/batchnorm/add/yБ
+model_2/batch_normalization_4/batchnorm/addAddV2>model_2/batch_normalization_4/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2-
+model_2/batch_normalization_4/batchnorm/addЊ
-model_2/batch_normalization_4/batchnorm/RsqrtRsqrt/model_2/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А2/
-model_2/batch_normalization_4/batchnorm/Rsqrtщ
:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOpю
+model_2/batch_normalization_4/batchnorm/mulMul1model_2/batch_normalization_4/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2-
+model_2/batch_normalization_4/batchnorm/mulю
-model_2/batch_normalization_4/batchnorm/mul_1Mul.model_2/token_and_position_embedding_4/add:z:0/model_2/batch_normalization_4/batchnorm/mul:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2/
-model_2/batch_normalization_4/batchnorm/mul_1у
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_1ю
-model_2/batch_normalization_4/batchnorm/mul_2Mul@model_2/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2/
-model_2/batch_normalization_4/batchnorm/mul_2у
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02:
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_2ь
+model_2/batch_normalization_4/batchnorm/subSub@model_2/batch_normalization_4/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2-
+model_2/batch_normalization_4/batchnorm/subГ
-model_2/batch_normalization_4/batchnorm/add_1AddV21model_2/batch_normalization_4/batchnorm/mul_1:z:0/model_2/batch_normalization_4/batchnorm/sub:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2/
-model_2/batch_normalization_4/batchnorm/add_1Ъ
*model_2/average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/average_pooling1d_4/ExpandDims/dimВ
&model_2/average_pooling1d_4/ExpandDims
ExpandDims1model_2/batch_normalization_4/batchnorm/add_1:z:03model_2/average_pooling1d_4/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€ДRА2(
&model_2/average_pooling1d_4/ExpandDims€
#model_2/average_pooling1d_4/AvgPoolAvgPool/model_2/average_pooling1d_4/ExpandDims:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
ksize	
И'*
paddingVALID*
strides	
И'2%
#model_2/average_pooling1d_4/AvgPool—
#model_2/average_pooling1d_4/SqueezeSqueeze,model_2/average_pooling1d_4/AvgPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€А*
squeeze_dims
2%
#model_2/average_pooling1d_4/Squeeze”
Umodel_2/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_9_multi_head_attention_9_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02W
Umodel_2/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpИ
Fmodel_2/transformer_block_9/multi_head_attention_9/query/einsum/EinsumEinsum,model_2/average_pooling1d_4/Squeeze:output:0]model_2/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2H
Fmodel_2/transformer_block_9/multi_head_attention_9/query/einsum/Einsum∞
Kmodel_2/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_9_multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes
:	А*
dtype02M
Kmodel_2/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpж
<model_2/transformer_block_9/multi_head_attention_9/query/addAddV2Omodel_2/transformer_block_9/multi_head_attention_9/query/einsum/Einsum:output:0Smodel_2/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2>
<model_2/transformer_block_9/multi_head_attention_9/query/addЌ
Smodel_2/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_2_transformer_block_9_multi_head_attention_9_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02U
Smodel_2/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpВ
Dmodel_2/transformer_block_9/multi_head_attention_9/key/einsum/EinsumEinsum,model_2/average_pooling1d_4/Squeeze:output:0[model_2/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2F
Dmodel_2/transformer_block_9/multi_head_attention_9/key/einsum/Einsum™
Imodel_2/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpReadVariableOpRmodel_2_transformer_block_9_multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes
:	А*
dtype02K
Imodel_2/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpё
:model_2/transformer_block_9/multi_head_attention_9/key/addAddV2Mmodel_2/transformer_block_9/multi_head_attention_9/key/einsum/Einsum:output:0Qmodel_2/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2<
:model_2/transformer_block_9/multi_head_attention_9/key/add”
Umodel_2/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_9_multi_head_attention_9_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02W
Umodel_2/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpИ
Fmodel_2/transformer_block_9/multi_head_attention_9/value/einsum/EinsumEinsum,model_2/average_pooling1d_4/Squeeze:output:0]model_2/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2H
Fmodel_2/transformer_block_9/multi_head_attention_9/value/einsum/Einsum∞
Kmodel_2/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_9_multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes
:	А*
dtype02M
Kmodel_2/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpж
<model_2/transformer_block_9/multi_head_attention_9/value/addAddV2Omodel_2/transformer_block_9/multi_head_attention_9/value/einsum/Einsum:output:0Smodel_2/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2>
<model_2/transformer_block_9/multi_head_attention_9/value/addє
8model_2/transformer_block_9/multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=2:
8model_2/transformer_block_9/multi_head_attention_9/Mul/yЈ
6model_2/transformer_block_9/multi_head_attention_9/MulMul@model_2/transformer_block_9/multi_head_attention_9/query/add:z:0Amodel_2/transformer_block_9/multi_head_attention_9/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А28
6model_2/transformer_block_9/multi_head_attention_9/Mulм
@model_2/transformer_block_9/multi_head_attention_9/einsum/EinsumEinsum>model_2/transformer_block_9/multi_head_attention_9/key/add:z:0:model_2/transformer_block_9/multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€*
equationaecd,abcd->acbe2B
@model_2/transformer_block_9/multi_head_attention_9/einsum/EinsumШ
Bmodel_2/transformer_block_9/multi_head_attention_9/softmax/SoftmaxSoftmaxImodel_2/transformer_block_9/multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€2D
Bmodel_2/transformer_block_9/multi_head_attention_9/softmax/SoftmaxЮ
Cmodel_2/transformer_block_9/multi_head_attention_9/dropout/IdentityIdentityLmodel_2/transformer_block_9/multi_head_attention_9/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€2E
Cmodel_2/transformer_block_9/multi_head_attention_9/dropout/IdentityЕ
Bmodel_2/transformer_block_9/multi_head_attention_9/einsum_1/EinsumEinsumLmodel_2/transformer_block_9/multi_head_attention_9/dropout/Identity:output:0@model_2/transformer_block_9/multi_head_attention_9/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationacbe,aecd->abcd2D
Bmodel_2/transformer_block_9/multi_head_attention_9/einsum_1/Einsumф
`model_2/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_2_transformer_block_9_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02b
`model_2/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpƒ
Qmodel_2/transformer_block_9/multi_head_attention_9/attention_output/einsum/EinsumEinsumKmodel_2/transformer_block_9/multi_head_attention_9/einsum_1/Einsum:output:0hmodel_2/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€А*
equationabcd,cde->abe2S
Qmodel_2/transformer_block_9/multi_head_attention_9/attention_output/einsum/EinsumЌ
Vmodel_2/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOp_model_2_transformer_block_9_multi_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype02X
Vmodel_2/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpО
Gmodel_2/transformer_block_9/multi_head_attention_9/attention_output/addAddV2Zmodel_2/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum:output:0^model_2/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2I
Gmodel_2/transformer_block_9/multi_head_attention_9/attention_output/addт
/model_2/transformer_block_9/dropout_22/IdentityIdentityKmodel_2/transformer_block_9/multi_head_attention_9/attention_output/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€А21
/model_2/transformer_block_9/dropout_22/Identityк
model_2/transformer_block_9/addAddV2,model_2/average_pooling1d_4/Squeeze:output:08model_2/transformer_block_9/dropout_22/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
model_2/transformer_block_9/addр
Qmodel_2/transformer_block_9/layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_2/transformer_block_9/layer_normalization_18/moments/mean/reduction_indices“
?model_2/transformer_block_9/layer_normalization_18/moments/meanMean#model_2/transformer_block_9/add:z:0Zmodel_2/transformer_block_9/layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2A
?model_2/transformer_block_9/layer_normalization_18/moments/meanҐ
Gmodel_2/transformer_block_9/layer_normalization_18/moments/StopGradientStopGradientHmodel_2/transformer_block_9/layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2I
Gmodel_2/transformer_block_9/layer_normalization_18/moments/StopGradientя
Lmodel_2/transformer_block_9/layer_normalization_18/moments/SquaredDifferenceSquaredDifference#model_2/transformer_block_9/add:z:0Pmodel_2/transformer_block_9/layer_normalization_18/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2N
Lmodel_2/transformer_block_9/layer_normalization_18/moments/SquaredDifferenceш
Umodel_2/transformer_block_9/layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_2/transformer_block_9/layer_normalization_18/moments/variance/reduction_indicesЛ
Cmodel_2/transformer_block_9/layer_normalization_18/moments/varianceMeanPmodel_2/transformer_block_9/layer_normalization_18/moments/SquaredDifference:z:0^model_2/transformer_block_9/layer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2E
Cmodel_2/transformer_block_9/layer_normalization_18/moments/varianceЌ
Bmodel_2/transformer_block_9/layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52D
Bmodel_2/transformer_block_9/layer_normalization_18/batchnorm/add/yё
@model_2/transformer_block_9/layer_normalization_18/batchnorm/addAddV2Lmodel_2/transformer_block_9/layer_normalization_18/moments/variance:output:0Kmodel_2/transformer_block_9/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2B
@model_2/transformer_block_9/layer_normalization_18/batchnorm/addН
Bmodel_2/transformer_block_9/layer_normalization_18/batchnorm/RsqrtRsqrtDmodel_2/transformer_block_9/layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2D
Bmodel_2/transformer_block_9/layer_normalization_18/batchnorm/RsqrtЄ
Omodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_2_transformer_block_9_layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02Q
Omodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpг
@model_2/transformer_block_9/layer_normalization_18/batchnorm/mulMulFmodel_2/transformer_block_9/layer_normalization_18/batchnorm/Rsqrt:y:0Wmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2B
@model_2/transformer_block_9/layer_normalization_18/batchnorm/mul±
Bmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul_1Mul#model_2/transformer_block_9/add:z:0Dmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2D
Bmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul_1÷
Bmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul_2MulHmodel_2/transformer_block_9/layer_normalization_18/moments/mean:output:0Dmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2D
Bmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul_2ђ
Kmodel_2/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOpTmodel_2_transformer_block_9_layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02M
Kmodel_2/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpя
@model_2/transformer_block_9/layer_normalization_18/batchnorm/subSubSmodel_2/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp:value:0Fmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2B
@model_2/transformer_block_9/layer_normalization_18/batchnorm/sub÷
Bmodel_2/transformer_block_9/layer_normalization_18/batchnorm/add_1AddV2Fmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul_1:z:0Dmodel_2/transformer_block_9/layer_normalization_18/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2D
Bmodel_2/transformer_block_9/layer_normalization_18/batchnorm/add_1Ѓ
Jmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpReadVariableOpSmodel_2_transformer_block_9_sequential_9_dense_24_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype02L
Jmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpќ
@model_2/transformer_block_9/sequential_9/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_2/transformer_block_9/sequential_9/dense_24/Tensordot/axes’
@model_2/transformer_block_9/sequential_9/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_2/transformer_block_9/sequential_9/dense_24/Tensordot/freeь
Amodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/ShapeShapeFmodel_2/transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/ShapeЎ
Imodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2/axisЋ
Dmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2GatherV2Jmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Shape:output:0Imodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/free:output:0Rmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2№
Kmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1/axis—
Fmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1GatherV2Jmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Shape:output:0Imodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/axes:output:0Tmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1–
Amodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Const»
@model_2/transformer_block_9/sequential_9/dense_24/Tensordot/ProdProdMmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2:output:0Jmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_2/transformer_block_9/sequential_9/dense_24/Tensordot/Prod‘
Cmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Const_1–
Bmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Prod_1ProdOmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1:output:0Lmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Prod_1‘
Gmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat/axis™
Bmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concatConcatV2Imodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/free:output:0Imodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/axes:output:0Pmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat‘
Amodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/stackPackImodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Prod:output:0Kmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/stackз
Emodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/transpose	TransposeFmodel_2/transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0Kmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2G
Emodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/transposeз
Cmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/ReshapeReshapeImodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/transpose:y:0Jmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2E
Cmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Reshapeз
Bmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/MatMulMatMulLmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Reshape:output:0Rmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2D
Bmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/MatMul’
Cmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2E
Cmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Const_2Ў
Imodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat_1/axisЈ
Dmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat_1ConcatV2Mmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2:output:0Lmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/Const_2:output:0Rmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat_1ў
;model_2/transformer_block_9/sequential_9/dense_24/TensordotReshapeLmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/MatMul:product:0Mmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2=
;model_2/transformer_block_9/sequential_9/dense_24/Tensordot£
Hmodel_2/transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_transformer_block_9_sequential_9_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02J
Hmodel_2/transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp–
9model_2/transformer_block_9/sequential_9/dense_24/BiasAddBiasAddDmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot:output:0Pmodel_2/transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2;
9model_2/transformer_block_9/sequential_9/dense_24/BiasAddу
6model_2/transformer_block_9/sequential_9/dense_24/ReluReluBmodel_2/transformer_block_9/sequential_9/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А28
6model_2/transformer_block_9/sequential_9/dense_24/ReluЃ
Jmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpReadVariableOpSmodel_2_transformer_block_9_sequential_9_dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype02L
Jmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpќ
@model_2/transformer_block_9/sequential_9/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_2/transformer_block_9/sequential_9/dense_25/Tensordot/axes’
@model_2/transformer_block_9/sequential_9/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_2/transformer_block_9/sequential_9/dense_25/Tensordot/freeъ
Amodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/ShapeShapeDmodel_2/transformer_block_9/sequential_9/dense_24/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/ShapeЎ
Imodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2/axisЋ
Dmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2GatherV2Jmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Shape:output:0Imodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/free:output:0Rmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2№
Kmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1/axis—
Fmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1GatherV2Jmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Shape:output:0Imodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/axes:output:0Tmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1–
Amodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Const»
@model_2/transformer_block_9/sequential_9/dense_25/Tensordot/ProdProdMmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2:output:0Jmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_2/transformer_block_9/sequential_9/dense_25/Tensordot/Prod‘
Cmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Const_1–
Bmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Prod_1ProdOmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1:output:0Lmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Prod_1‘
Gmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat/axis™
Bmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concatConcatV2Imodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/free:output:0Imodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/axes:output:0Pmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat‘
Amodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/stackPackImodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Prod:output:0Kmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/stackе
Emodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/transpose	TransposeDmodel_2/transformer_block_9/sequential_9/dense_24/Relu:activations:0Kmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2G
Emodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/transposeз
Cmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/ReshapeReshapeImodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/transpose:y:0Jmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2E
Cmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Reshapeз
Bmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/MatMulMatMulLmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Reshape:output:0Rmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2D
Bmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/MatMul’
Cmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2E
Cmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Const_2Ў
Imodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat_1/axisЈ
Dmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat_1ConcatV2Mmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2:output:0Lmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/Const_2:output:0Rmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat_1ў
;model_2/transformer_block_9/sequential_9/dense_25/TensordotReshapeLmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/MatMul:product:0Mmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2=
;model_2/transformer_block_9/sequential_9/dense_25/Tensordot£
Hmodel_2/transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_transformer_block_9_sequential_9_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02J
Hmodel_2/transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp–
9model_2/transformer_block_9/sequential_9/dense_25/BiasAddBiasAddDmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot:output:0Pmodel_2/transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2;
9model_2/transformer_block_9/sequential_9/dense_25/BiasAddй
/model_2/transformer_block_9/dropout_23/IdentityIdentityBmodel_2/transformer_block_9/sequential_9/dense_25/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А21
/model_2/transformer_block_9/dropout_23/IdentityИ
!model_2/transformer_block_9/add_1AddV2Fmodel_2/transformer_block_9/layer_normalization_18/batchnorm/add_1:z:08model_2/transformer_block_9/dropout_23/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2#
!model_2/transformer_block_9/add_1р
Qmodel_2/transformer_block_9/layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_2/transformer_block_9/layer_normalization_19/moments/mean/reduction_indices‘
?model_2/transformer_block_9/layer_normalization_19/moments/meanMean%model_2/transformer_block_9/add_1:z:0Zmodel_2/transformer_block_9/layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2A
?model_2/transformer_block_9/layer_normalization_19/moments/meanҐ
Gmodel_2/transformer_block_9/layer_normalization_19/moments/StopGradientStopGradientHmodel_2/transformer_block_9/layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2I
Gmodel_2/transformer_block_9/layer_normalization_19/moments/StopGradientб
Lmodel_2/transformer_block_9/layer_normalization_19/moments/SquaredDifferenceSquaredDifference%model_2/transformer_block_9/add_1:z:0Pmodel_2/transformer_block_9/layer_normalization_19/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2N
Lmodel_2/transformer_block_9/layer_normalization_19/moments/SquaredDifferenceш
Umodel_2/transformer_block_9/layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_2/transformer_block_9/layer_normalization_19/moments/variance/reduction_indicesЛ
Cmodel_2/transformer_block_9/layer_normalization_19/moments/varianceMeanPmodel_2/transformer_block_9/layer_normalization_19/moments/SquaredDifference:z:0^model_2/transformer_block_9/layer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2E
Cmodel_2/transformer_block_9/layer_normalization_19/moments/varianceЌ
Bmodel_2/transformer_block_9/layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52D
Bmodel_2/transformer_block_9/layer_normalization_19/batchnorm/add/yё
@model_2/transformer_block_9/layer_normalization_19/batchnorm/addAddV2Lmodel_2/transformer_block_9/layer_normalization_19/moments/variance:output:0Kmodel_2/transformer_block_9/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2B
@model_2/transformer_block_9/layer_normalization_19/batchnorm/addН
Bmodel_2/transformer_block_9/layer_normalization_19/batchnorm/RsqrtRsqrtDmodel_2/transformer_block_9/layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2D
Bmodel_2/transformer_block_9/layer_normalization_19/batchnorm/RsqrtЄ
Omodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_2_transformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02Q
Omodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpг
@model_2/transformer_block_9/layer_normalization_19/batchnorm/mulMulFmodel_2/transformer_block_9/layer_normalization_19/batchnorm/Rsqrt:y:0Wmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2B
@model_2/transformer_block_9/layer_normalization_19/batchnorm/mul≥
Bmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul_1Mul%model_2/transformer_block_9/add_1:z:0Dmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2D
Bmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul_1÷
Bmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul_2MulHmodel_2/transformer_block_9/layer_normalization_19/moments/mean:output:0Dmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2D
Bmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul_2ђ
Kmodel_2/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOpTmodel_2_transformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02M
Kmodel_2/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpя
@model_2/transformer_block_9/layer_normalization_19/batchnorm/subSubSmodel_2/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp:value:0Fmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2B
@model_2/transformer_block_9/layer_normalization_19/batchnorm/sub÷
Bmodel_2/transformer_block_9/layer_normalization_19/batchnorm/add_1AddV2Fmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul_1:z:0Dmodel_2/transformer_block_9/layer_normalization_19/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2D
Bmodel_2/transformer_block_9/layer_normalization_19/batchnorm/add_1Г
model_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
model_2/flatten_2/Constё
model_2/flatten_2/ReshapeReshapeFmodel_2/transformer_block_9/layer_normalization_19/batchnorm/add_1:z:0 model_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_2/flatten_2/ReshapeИ
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_2/concatenate_2/concat/axisё
model_2/concatenate_2/concatConcatV2"model_2/flatten_2/Reshape:output:0input_10*model_2/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€И2
model_2/concatenate_2/concatЅ
&model_2/dense_26/MatMul/ReadVariableOpReadVariableOp/model_2_dense_26_matmul_readvariableop_resource*
_output_shapes
:	И@*
dtype02(
&model_2/dense_26/MatMul/ReadVariableOp≈
model_2/dense_26/MatMulMatMul%model_2/concatenate_2/concat:output:0.model_2/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_2/dense_26/MatMulњ
'model_2/dense_26/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/dense_26/BiasAdd/ReadVariableOp≈
model_2/dense_26/BiasAddBiasAdd!model_2/dense_26/MatMul:product:0/model_2/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_2/dense_26/BiasAddЛ
model_2/dense_26/ReluRelu!model_2/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_2/dense_26/ReluЭ
model_2/dropout_24/IdentityIdentity#model_2/dense_26/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_2/dropout_24/Identityј
&model_2/dense_27/MatMul/ReadVariableOpReadVariableOp/model_2_dense_27_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_2/dense_27/MatMul/ReadVariableOpƒ
model_2/dense_27/MatMulMatMul$model_2/dropout_24/Identity:output:0.model_2/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_2/dense_27/MatMulњ
'model_2/dense_27/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/dense_27/BiasAdd/ReadVariableOp≈
model_2/dense_27/BiasAddBiasAdd!model_2/dense_27/MatMul:product:0/model_2/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_2/dense_27/BiasAddЛ
model_2/dense_27/ReluRelu!model_2/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_2/dense_27/ReluЭ
model_2/dropout_25/IdentityIdentity#model_2/dense_27/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_2/dropout_25/Identityј
&model_2/dense_28/MatMul/ReadVariableOpReadVariableOp/model_2_dense_28_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_2/dense_28/MatMul/ReadVariableOpƒ
model_2/dense_28/MatMulMatMul$model_2/dropout_25/Identity:output:0.model_2/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_2/dense_28/MatMulњ
'model_2/dense_28/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_2/dense_28/BiasAdd/ReadVariableOp≈
model_2/dense_28/BiasAddBiasAdd!model_2/dense_28/MatMul:product:0/model_2/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_2/dense_28/BiasAddА
IdentityIdentity!model_2/dense_28/BiasAdd:output:07^model_2/batch_normalization_4/batchnorm/ReadVariableOp9^model_2/batch_normalization_4/batchnorm/ReadVariableOp_19^model_2/batch_normalization_4/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp(^model_2/dense_26/BiasAdd/ReadVariableOp'^model_2/dense_26/MatMul/ReadVariableOp(^model_2/dense_27/BiasAdd/ReadVariableOp'^model_2/dense_27/MatMul/ReadVariableOp(^model_2/dense_28/BiasAdd/ReadVariableOp'^model_2/dense_28/MatMul/ReadVariableOpD^model_2/token_and_position_embedding_4/embedding_8/embedding_lookupD^model_2/token_and_position_embedding_4/embedding_9/embedding_lookupL^model_2/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpP^model_2/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpL^model_2/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpP^model_2/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpW^model_2/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpa^model_2/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpJ^model_2/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpT^model_2/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpV^model_2/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpV^model_2/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpI^model_2/transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOpK^model_2/transformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpI^model_2/transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOpK^model_2/transformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::2p
6model_2/batch_normalization_4/batchnorm/ReadVariableOp6model_2/batch_normalization_4/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_18model_2/batch_normalization_4/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_28model_2/batch_normalization_4/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp2R
'model_2/dense_26/BiasAdd/ReadVariableOp'model_2/dense_26/BiasAdd/ReadVariableOp2P
&model_2/dense_26/MatMul/ReadVariableOp&model_2/dense_26/MatMul/ReadVariableOp2R
'model_2/dense_27/BiasAdd/ReadVariableOp'model_2/dense_27/BiasAdd/ReadVariableOp2P
&model_2/dense_27/MatMul/ReadVariableOp&model_2/dense_27/MatMul/ReadVariableOp2R
'model_2/dense_28/BiasAdd/ReadVariableOp'model_2/dense_28/BiasAdd/ReadVariableOp2P
&model_2/dense_28/MatMul/ReadVariableOp&model_2/dense_28/MatMul/ReadVariableOp2К
Cmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookupCmodel_2/token_and_position_embedding_4/embedding_8/embedding_lookup2К
Cmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookupCmodel_2/token_and_position_embedding_4/embedding_9/embedding_lookup2Ъ
Kmodel_2/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpKmodel_2/transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp2Ґ
Omodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpOmodel_2/transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp2Ъ
Kmodel_2/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpKmodel_2/transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp2Ґ
Omodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpOmodel_2/transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp2∞
Vmodel_2/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpVmodel_2/transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp2ƒ
`model_2/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp`model_2/transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2Ц
Imodel_2/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpImodel_2/transformer_block_9/multi_head_attention_9/key/add/ReadVariableOp2™
Smodel_2/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpSmodel_2/transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2Ъ
Kmodel_2/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpKmodel_2/transformer_block_9/multi_head_attention_9/query/add/ReadVariableOp2Ѓ
Umodel_2/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpUmodel_2/transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2Ъ
Kmodel_2/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpKmodel_2/transformer_block_9/multi_head_attention_9/value/add/ReadVariableOp2Ѓ
Umodel_2/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpUmodel_2/transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2Ф
Hmodel_2/transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOpHmodel_2/transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp2Ш
Jmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpJmodel_2/transformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOp2Ф
Hmodel_2/transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOpHmodel_2/transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp2Ш
Jmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpJmodel_2/transformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_9:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
input_10
»
c
E__inference_dropout_24_layer_call_and_return_conditional_losses_43585

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ґ
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_45210

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*+
_input_shapes
:€€€€€€€€€А:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
»
c
E__inference_dropout_25_layer_call_and_return_conditional_losses_43642

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
»
c
E__inference_dropout_25_layer_call_and_return_conditional_losses_45312

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ыч
ы
B__inference_model_2_layer_call_and_return_conditional_losses_44340
inputs_0
inputs_1E
Atoken_and_position_embedding_4_embedding_9_embedding_lookup_44107E
Atoken_and_position_embedding_4_embedding_8_embedding_lookup_44113/
+batch_normalization_4_assignmovingavg_441271
-batch_normalization_4_assignmovingavg_1_44133?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resourceZ
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
Ktransformer_block_9_sequential_9_dense_24_tensordot_readvariableop_resourceM
Itransformer_block_9_sequential_9_dense_24_biasadd_readvariableop_resourceO
Ktransformer_block_9_sequential_9_dense_25_tensordot_readvariableop_resourceM
Itransformer_block_9_sequential_9_dense_25_biasadd_readvariableop_resourceT
Ptransformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource
identityИҐ9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_4/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_4/batchnorm/ReadVariableOpҐ2batch_normalization_4/batchnorm/mul/ReadVariableOpҐdense_26/BiasAdd/ReadVariableOpҐdense_26/MatMul/ReadVariableOpҐdense_27/BiasAdd/ReadVariableOpҐdense_27/MatMul/ReadVariableOpҐdense_28/BiasAdd/ReadVariableOpҐdense_28/MatMul/ReadVariableOpҐ;token_and_position_embedding_4/embedding_8/embedding_lookupҐ;token_and_position_embedding_4/embedding_9/embedding_lookupҐCtransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpҐGtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpҐCtransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpҐGtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpҐNtransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpҐXtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpҐAtransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpҐKtransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpҐCtransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpҐMtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpҐCtransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpҐMtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpҐ@transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOpҐBtransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpҐ@transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOpҐBtransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpД
$token_and_position_embedding_4/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_4/Shapeї
2token_and_position_embedding_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€24
2token_and_position_embedding_4/strided_slice/stackґ
4token_and_position_embedding_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_4/strided_slice/stack_1ґ
4token_and_position_embedding_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_4/strided_slice/stack_2Ь
,token_and_position_embedding_4/strided_sliceStridedSlice-token_and_position_embedding_4/Shape:output:0;token_and_position_embedding_4/strided_slice/stack:output:0=token_and_position_embedding_4/strided_slice/stack_1:output:0=token_and_position_embedding_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_4/strided_sliceЪ
*token_and_position_embedding_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_4/range/startЪ
*token_and_position_embedding_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_4/range/deltaЫ
$token_and_position_embedding_4/rangeRange3token_and_position_embedding_4/range/start:output:05token_and_position_embedding_4/strided_slice:output:03token_and_position_embedding_4/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2&
$token_and_position_embedding_4/range…
;token_and_position_embedding_4/embedding_9/embedding_lookupResourceGatherAtoken_and_position_embedding_4_embedding_9_embedding_lookup_44107-token_and_position_embedding_4/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_4/embedding_9/embedding_lookup/44107*(
_output_shapes
:€€€€€€€€€А*
dtype02=
;token_and_position_embedding_4/embedding_9/embedding_lookupХ
Dtoken_and_position_embedding_4/embedding_9/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_4/embedding_9/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_4/embedding_9/embedding_lookup/44107*(
_output_shapes
:€€€€€€€€€А2F
Dtoken_and_position_embedding_4/embedding_9/embedding_lookup/IdentityЮ
Ftoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2H
Ftoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1ґ
/token_and_position_embedding_4/embedding_8/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€ДR21
/token_and_position_embedding_4/embedding_8/Cast‘
;token_and_position_embedding_4/embedding_8/embedding_lookupResourceGatherAtoken_and_position_embedding_4_embedding_8_embedding_lookup_441133token_and_position_embedding_4/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*T
_classJ
HFloc:@token_and_position_embedding_4/embedding_8/embedding_lookup/44113*-
_output_shapes
:€€€€€€€€€ДRА*
dtype02=
;token_and_position_embedding_4/embedding_8/embedding_lookupЪ
Dtoken_and_position_embedding_4/embedding_8/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_4/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*T
_classJ
HFloc:@token_and_position_embedding_4/embedding_8/embedding_lookup/44113*-
_output_shapes
:€€€€€€€€€ДRА2F
Dtoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity£
Ftoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2H
Ftoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1Ђ
"token_and_position_embedding_4/addAddV2Otoken_and_position_embedding_4/embedding_8/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_4/embedding_9/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2$
"token_and_position_embedding_4/addљ
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_4/moments/mean/reduction_indicesц
"batch_normalization_4/moments/meanMean&token_and_position_embedding_4/add:z:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2$
"batch_normalization_4/moments/mean√
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*#
_output_shapes
:А2,
*batch_normalization_4/moments/StopGradientМ
/batch_normalization_4/moments/SquaredDifferenceSquaredDifference&token_and_position_embedding_4/add:z:03batch_normalization_4/moments/StopGradient:output:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА21
/batch_normalization_4/moments/SquaredDifference≈
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_4/moments/variance/reduction_indicesП
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2(
&batch_normalization_4/moments/varianceƒ
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeezeћ
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1Н
+batch_normalization_4/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/44127*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_4/AssignMovingAvg/decay’
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_4_assignmovingavg_44127*
_output_shapes	
:А*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpя
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/44127*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/sub÷
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/44127*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/mul±
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_4_assignmovingavg_44127-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/44127*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpУ
-batch_normalization_4/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/44133*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_4/AssignMovingAvg_1/decayџ
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_4_assignmovingavg_1_44133*
_output_shapes	
:А*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpй
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/44133*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/subа
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/44133*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/mulљ
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_4_assignmovingavg_1_44133/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/44133*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/yџ
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/add¶
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/Rsqrtб
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOpё
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/mulё
%batch_normalization_4/batchnorm/mul_1Mul&token_and_position_embedding_4/add:z:0'batch_normalization_4/batchnorm/mul:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2'
%batch_normalization_4/batchnorm/mul_1‘
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_4/batchnorm/mul_2’
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpЏ
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_4/batchnorm/subг
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*-
_output_shapes
:€€€€€€€€€ДRА2'
%batch_normalization_4/batchnorm/add_1К
"average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_4/ExpandDims/dimв
average_pooling1d_4/ExpandDims
ExpandDims)batch_normalization_4/batchnorm/add_1:z:0+average_pooling1d_4/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€ДRА2 
average_pooling1d_4/ExpandDimsз
average_pooling1d_4/AvgPoolAvgPool'average_pooling1d_4/ExpandDims:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
ksize	
И'*
paddingVALID*
strides	
И'2
average_pooling1d_4/AvgPoolє
average_pooling1d_4/SqueezeSqueeze$average_pooling1d_4/AvgPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€А*
squeeze_dims
2
average_pooling1d_4/Squeezeї
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_9_multi_head_attention_9_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02O
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpи
>transformer_block_9/multi_head_attention_9/query/einsum/EinsumEinsum$average_pooling1d_4/Squeeze:output:0Utransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2@
>transformer_block_9/multi_head_attention_9/query/einsum/EinsumШ
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpReadVariableOpLtransformer_block_9_multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes
:	А*
dtype02E
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp∆
4transformer_block_9/multi_head_attention_9/query/addAddV2Gtransformer_block_9/multi_head_attention_9/query/einsum/Einsum:output:0Ktransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А26
4transformer_block_9/multi_head_attention_9/query/addµ
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_9_multi_head_attention_9_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02M
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpв
<transformer_block_9/multi_head_attention_9/key/einsum/EinsumEinsum$average_pooling1d_4/Squeeze:output:0Stransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2>
<transformer_block_9/multi_head_attention_9/key/einsum/EinsumТ
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpReadVariableOpJtransformer_block_9_multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes
:	А*
dtype02C
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpЊ
2transformer_block_9/multi_head_attention_9/key/addAddV2Etransformer_block_9/multi_head_attention_9/key/einsum/Einsum:output:0Itransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А24
2transformer_block_9/multi_head_attention_9/key/addї
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_9_multi_head_attention_9_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02O
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpи
>transformer_block_9/multi_head_attention_9/value/einsum/EinsumEinsum$average_pooling1d_4/Squeeze:output:0Utransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2@
>transformer_block_9/multi_head_attention_9/value/einsum/EinsumШ
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpReadVariableOpLtransformer_block_9_multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes
:	А*
dtype02E
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp∆
4transformer_block_9/multi_head_attention_9/value/addAddV2Gtransformer_block_9/multi_head_attention_9/value/einsum/Einsum:output:0Ktransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А26
4transformer_block_9/multi_head_attention_9/value/add©
0transformer_block_9/multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=22
0transformer_block_9/multi_head_attention_9/Mul/yЧ
.transformer_block_9/multi_head_attention_9/MulMul8transformer_block_9/multi_head_attention_9/query/add:z:09transformer_block_9/multi_head_attention_9/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А20
.transformer_block_9/multi_head_attention_9/Mulћ
8transformer_block_9/multi_head_attention_9/einsum/EinsumEinsum6transformer_block_9/multi_head_attention_9/key/add:z:02transformer_block_9/multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€*
equationaecd,abcd->acbe2:
8transformer_block_9/multi_head_attention_9/einsum/EinsumА
:transformer_block_9/multi_head_attention_9/softmax/SoftmaxSoftmaxAtransformer_block_9/multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€2<
:transformer_block_9/multi_head_attention_9/softmax/Softmax…
@transformer_block_9/multi_head_attention_9/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2B
@transformer_block_9/multi_head_attention_9/dropout/dropout/Const“
>transformer_block_9/multi_head_attention_9/dropout/dropout/MulMulDtransformer_block_9/multi_head_attention_9/softmax/Softmax:softmax:0Itransformer_block_9/multi_head_attention_9/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€2@
>transformer_block_9/multi_head_attention_9/dropout/dropout/Mulш
@transformer_block_9/multi_head_attention_9/dropout/dropout/ShapeShapeDtransformer_block_9/multi_head_attention_9/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_9/multi_head_attention_9/dropout/dropout/Shapeб
Wtransformer_block_9/multi_head_attention_9/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_9/multi_head_attention_9/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
dtype0*

seed2Y
Wtransformer_block_9/multi_head_attention_9/dropout/dropout/random_uniform/RandomUniformџ
Itransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqual/yТ
Gtransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_9/multi_head_attention_9/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€2I
Gtransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqual†
?transformer_block_9/multi_head_attention_9/dropout/dropout/CastCastKtransformer_block_9/multi_head_attention_9/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€2A
?transformer_block_9/multi_head_attention_9/dropout/dropout/Castќ
@transformer_block_9/multi_head_attention_9/dropout/dropout/Mul_1MulBtransformer_block_9/multi_head_attention_9/dropout/dropout/Mul:z:0Ctransformer_block_9/multi_head_attention_9/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€2B
@transformer_block_9/multi_head_attention_9/dropout/dropout/Mul_1е
:transformer_block_9/multi_head_attention_9/einsum_1/EinsumEinsumDtransformer_block_9/multi_head_attention_9/dropout/dropout/Mul_1:z:08transformer_block_9/multi_head_attention_9/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationacbe,aecd->abcd2<
:transformer_block_9/multi_head_attention_9/einsum_1/Einsum№
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_9_multi_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02Z
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp§
Itransformer_block_9/multi_head_attention_9/attention_output/einsum/EinsumEinsumCtransformer_block_9/multi_head_attention_9/einsum_1/Einsum:output:0`transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€А*
equationabcd,cde->abe2K
Itransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsumµ
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_9_multi_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype02P
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpо
?transformer_block_9/multi_head_attention_9/attention_output/addAddV2Rtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum:output:0Vtransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2A
?transformer_block_9/multi_head_attention_9/attention_output/add°
,transformer_block_9/dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2.
,transformer_block_9/dropout_22/dropout/ConstТ
*transformer_block_9/dropout_22/dropout/MulMulCtransformer_block_9/multi_head_attention_9/attention_output/add:z:05transformer_block_9/dropout_22/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2,
*transformer_block_9/dropout_22/dropout/Mulѕ
,transformer_block_9/dropout_22/dropout/ShapeShapeCtransformer_block_9/multi_head_attention_9/attention_output/add:z:0*
T0*
_output_shapes
:2.
,transformer_block_9/dropout_22/dropout/Shapeѓ
Ctransformer_block_9/dropout_22/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_9/dropout_22/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А*
dtype0*

seed*
seed22E
Ctransformer_block_9/dropout_22/dropout/random_uniform/RandomUniform≥
5transformer_block_9/dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=27
5transformer_block_9/dropout_22/dropout/GreaterEqual/yњ
3transformer_block_9/dropout_22/dropout/GreaterEqualGreaterEqualLtransformer_block_9/dropout_22/dropout/random_uniform/RandomUniform:output:0>transformer_block_9/dropout_22/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€А25
3transformer_block_9/dropout_22/dropout/GreaterEqualб
+transformer_block_9/dropout_22/dropout/CastCast7transformer_block_9/dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€А2-
+transformer_block_9/dropout_22/dropout/Castы
,transformer_block_9/dropout_22/dropout/Mul_1Mul.transformer_block_9/dropout_22/dropout/Mul:z:0/transformer_block_9/dropout_22/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€А2.
,transformer_block_9/dropout_22/dropout/Mul_1 
transformer_block_9/addAddV2$average_pooling1d_4/Squeeze:output:00transformer_block_9/dropout_22/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2
transformer_block_9/addа
Itransformer_block_9/layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_9/layer_normalization_18/moments/mean/reduction_indices≤
7transformer_block_9/layer_normalization_18/moments/meanMeantransformer_block_9/add:z:0Rtransformer_block_9/layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(29
7transformer_block_9/layer_normalization_18/moments/meanК
?transformer_block_9/layer_normalization_18/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2A
?transformer_block_9/layer_normalization_18/moments/StopGradientњ
Dtransformer_block_9/layer_normalization_18/moments/SquaredDifferenceSquaredDifferencetransformer_block_9/add:z:0Htransformer_block_9/layer_normalization_18/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2F
Dtransformer_block_9/layer_normalization_18/moments/SquaredDifferenceи
Mtransformer_block_9/layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_9/layer_normalization_18/moments/variance/reduction_indicesл
;transformer_block_9/layer_normalization_18/moments/varianceMeanHtransformer_block_9/layer_normalization_18/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2=
;transformer_block_9/layer_normalization_18/moments/varianceљ
:transformer_block_9/layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52<
:transformer_block_9/layer_normalization_18/batchnorm/add/yЊ
8transformer_block_9/layer_normalization_18/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_18/moments/variance:output:0Ctransformer_block_9/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2:
8transformer_block_9/layer_normalization_18/batchnorm/addх
:transformer_block_9/layer_normalization_18/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2<
:transformer_block_9/layer_normalization_18/batchnorm/Rsqrt†
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02I
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp√
8transformer_block_9/layer_normalization_18/batchnorm/mulMul>transformer_block_9/layer_normalization_18/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2:
8transformer_block_9/layer_normalization_18/batchnorm/mulС
:transformer_block_9/layer_normalization_18/batchnorm/mul_1Multransformer_block_9/add:z:0<transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_18/batchnorm/mul_1ґ
:transformer_block_9/layer_normalization_18/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_18/moments/mean:output:0<transformer_block_9/layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_18/batchnorm/mul_2Ф
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02E
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpњ
8transformer_block_9/layer_normalization_18/batchnorm/subSubKtransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_18/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2:
8transformer_block_9/layer_normalization_18/batchnorm/subґ
:transformer_block_9/layer_normalization_18/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_18/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_18/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_18/batchnorm/add_1Ц
Btransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_24_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype02D
Btransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpЊ
8transformer_block_9/sequential_9/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_9/sequential_9/dense_24/Tensordot/axes≈
8transformer_block_9/sequential_9/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_9/sequential_9/dense_24/Tensordot/freeд
9transformer_block_9/sequential_9/dense_24/Tensordot/ShapeShape>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_24/Tensordot/Shape»
Atransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2/axis£
<transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_24/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_24/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2ћ
Ctransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1/axis©
>transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_24/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_24/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1ј
9transformer_block_9/sequential_9/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_9/sequential_9/dense_24/Tensordot/Const®
8transformer_block_9/sequential_9/dense_24/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_9/sequential_9/dense_24/Tensordot/Prodƒ
;transformer_block_9/sequential_9/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_9/sequential_9/dense_24/Tensordot/Const_1∞
:transformer_block_9/sequential_9/dense_24/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_9/sequential_9/dense_24/Tensordot/Prod_1ƒ
?transformer_block_9/sequential_9/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_9/sequential_9/dense_24/Tensordot/concat/axisВ
:transformer_block_9/sequential_9/dense_24/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_24/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_24/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_9/sequential_9/dense_24/Tensordot/concatі
9transformer_block_9/sequential_9/dense_24/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_24/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_24/Tensordot/stack«
=transformer_block_9/sequential_9/dense_24/Tensordot/transpose	Transpose>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:0Ctransformer_block_9/sequential_9/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2?
=transformer_block_9/sequential_9/dense_24/Tensordot/transpose«
;transformer_block_9/sequential_9/dense_24/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_24/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2=
;transformer_block_9/sequential_9/dense_24/Tensordot/Reshape«
:transformer_block_9/sequential_9/dense_24/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_24/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/sequential_9/dense_24/Tensordot/MatMul≈
;transformer_block_9/sequential_9/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2=
;transformer_block_9/sequential_9/dense_24/Tensordot/Const_2»
Atransformer_block_9/sequential_9/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_24/Tensordot/concat_1/axisП
<transformer_block_9/sequential_9/dense_24/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_24/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_24/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_24/Tensordot/concat_1є
3transformer_block_9/sequential_9/dense_24/TensordotReshapeDtransformer_block_9/sequential_9/dense_24/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А25
3transformer_block_9/sequential_9/dense_24/TensordotЛ
@transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02B
@transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp∞
1transformer_block_9/sequential_9/dense_24/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_24/Tensordot:output:0Htransformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А23
1transformer_block_9/sequential_9/dense_24/BiasAddџ
.transformer_block_9/sequential_9/dense_24/ReluRelu:transformer_block_9/sequential_9/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А20
.transformer_block_9/sequential_9/dense_24/ReluЦ
Btransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_9_sequential_9_dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype02D
Btransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpЊ
8transformer_block_9/sequential_9/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_9/sequential_9/dense_25/Tensordot/axes≈
8transformer_block_9/sequential_9/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_9/sequential_9/dense_25/Tensordot/freeв
9transformer_block_9/sequential_9/dense_25/Tensordot/ShapeShape<transformer_block_9/sequential_9/dense_24/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_25/Tensordot/Shape»
Atransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2/axis£
<transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2GatherV2Btransformer_block_9/sequential_9/dense_25/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_25/Tensordot/free:output:0Jtransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2ћ
Ctransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1/axis©
>transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1GatherV2Btransformer_block_9/sequential_9/dense_25/Tensordot/Shape:output:0Atransformer_block_9/sequential_9/dense_25/Tensordot/axes:output:0Ltransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1ј
9transformer_block_9/sequential_9/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_9/sequential_9/dense_25/Tensordot/Const®
8transformer_block_9/sequential_9/dense_25/Tensordot/ProdProdEtransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2:output:0Btransformer_block_9/sequential_9/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_9/sequential_9/dense_25/Tensordot/Prodƒ
;transformer_block_9/sequential_9/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_9/sequential_9/dense_25/Tensordot/Const_1∞
:transformer_block_9/sequential_9/dense_25/Tensordot/Prod_1ProdGtransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2_1:output:0Dtransformer_block_9/sequential_9/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_9/sequential_9/dense_25/Tensordot/Prod_1ƒ
?transformer_block_9/sequential_9/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_9/sequential_9/dense_25/Tensordot/concat/axisВ
:transformer_block_9/sequential_9/dense_25/Tensordot/concatConcatV2Atransformer_block_9/sequential_9/dense_25/Tensordot/free:output:0Atransformer_block_9/sequential_9/dense_25/Tensordot/axes:output:0Htransformer_block_9/sequential_9/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_9/sequential_9/dense_25/Tensordot/concatі
9transformer_block_9/sequential_9/dense_25/Tensordot/stackPackAtransformer_block_9/sequential_9/dense_25/Tensordot/Prod:output:0Ctransformer_block_9/sequential_9/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_9/sequential_9/dense_25/Tensordot/stack≈
=transformer_block_9/sequential_9/dense_25/Tensordot/transpose	Transpose<transformer_block_9/sequential_9/dense_24/Relu:activations:0Ctransformer_block_9/sequential_9/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2?
=transformer_block_9/sequential_9/dense_25/Tensordot/transpose«
;transformer_block_9/sequential_9/dense_25/Tensordot/ReshapeReshapeAtransformer_block_9/sequential_9/dense_25/Tensordot/transpose:y:0Btransformer_block_9/sequential_9/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2=
;transformer_block_9/sequential_9/dense_25/Tensordot/Reshape«
:transformer_block_9/sequential_9/dense_25/Tensordot/MatMulMatMulDtransformer_block_9/sequential_9/dense_25/Tensordot/Reshape:output:0Jtransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/sequential_9/dense_25/Tensordot/MatMul≈
;transformer_block_9/sequential_9/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2=
;transformer_block_9/sequential_9/dense_25/Tensordot/Const_2»
Atransformer_block_9/sequential_9/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_9/sequential_9/dense_25/Tensordot/concat_1/axisП
<transformer_block_9/sequential_9/dense_25/Tensordot/concat_1ConcatV2Etransformer_block_9/sequential_9/dense_25/Tensordot/GatherV2:output:0Dtransformer_block_9/sequential_9/dense_25/Tensordot/Const_2:output:0Jtransformer_block_9/sequential_9/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_9/sequential_9/dense_25/Tensordot/concat_1є
3transformer_block_9/sequential_9/dense_25/TensordotReshapeDtransformer_block_9/sequential_9/dense_25/Tensordot/MatMul:product:0Etransformer_block_9/sequential_9/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А25
3transformer_block_9/sequential_9/dense_25/TensordotЛ
@transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_9_sequential_9_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02B
@transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp∞
1transformer_block_9/sequential_9/dense_25/BiasAddBiasAdd<transformer_block_9/sequential_9/dense_25/Tensordot:output:0Htransformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А23
1transformer_block_9/sequential_9/dense_25/BiasAdd°
,transformer_block_9/dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2.
,transformer_block_9/dropout_23/dropout/ConstЙ
*transformer_block_9/dropout_23/dropout/MulMul:transformer_block_9/sequential_9/dense_25/BiasAdd:output:05transformer_block_9/dropout_23/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2,
*transformer_block_9/dropout_23/dropout/Mul∆
,transformer_block_9/dropout_23/dropout/ShapeShape:transformer_block_9/sequential_9/dense_25/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_9/dropout_23/dropout/Shapeѓ
Ctransformer_block_9/dropout_23/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_9/dropout_23/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А*
dtype0*

seed*
seed22E
Ctransformer_block_9/dropout_23/dropout/random_uniform/RandomUniform≥
5transformer_block_9/dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=27
5transformer_block_9/dropout_23/dropout/GreaterEqual/yњ
3transformer_block_9/dropout_23/dropout/GreaterEqualGreaterEqualLtransformer_block_9/dropout_23/dropout/random_uniform/RandomUniform:output:0>transformer_block_9/dropout_23/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€А25
3transformer_block_9/dropout_23/dropout/GreaterEqualб
+transformer_block_9/dropout_23/dropout/CastCast7transformer_block_9/dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€А2-
+transformer_block_9/dropout_23/dropout/Castы
,transformer_block_9/dropout_23/dropout/Mul_1Mul.transformer_block_9/dropout_23/dropout/Mul:z:0/transformer_block_9/dropout_23/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€А2.
,transformer_block_9/dropout_23/dropout/Mul_1и
transformer_block_9/add_1AddV2>transformer_block_9/layer_normalization_18/batchnorm/add_1:z:00transformer_block_9/dropout_23/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2
transformer_block_9/add_1а
Itransformer_block_9/layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_9/layer_normalization_19/moments/mean/reduction_indicesі
7transformer_block_9/layer_normalization_19/moments/meanMeantransformer_block_9/add_1:z:0Rtransformer_block_9/layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(29
7transformer_block_9/layer_normalization_19/moments/meanК
?transformer_block_9/layer_normalization_19/moments/StopGradientStopGradient@transformer_block_9/layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2A
?transformer_block_9/layer_normalization_19/moments/StopGradientЅ
Dtransformer_block_9/layer_normalization_19/moments/SquaredDifferenceSquaredDifferencetransformer_block_9/add_1:z:0Htransformer_block_9/layer_normalization_19/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2F
Dtransformer_block_9/layer_normalization_19/moments/SquaredDifferenceи
Mtransformer_block_9/layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_9/layer_normalization_19/moments/variance/reduction_indicesл
;transformer_block_9/layer_normalization_19/moments/varianceMeanHtransformer_block_9/layer_normalization_19/moments/SquaredDifference:z:0Vtransformer_block_9/layer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2=
;transformer_block_9/layer_normalization_19/moments/varianceљ
:transformer_block_9/layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52<
:transformer_block_9/layer_normalization_19/batchnorm/add/yЊ
8transformer_block_9/layer_normalization_19/batchnorm/addAddV2Dtransformer_block_9/layer_normalization_19/moments/variance:output:0Ctransformer_block_9/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2:
8transformer_block_9/layer_normalization_19/batchnorm/addх
:transformer_block_9/layer_normalization_19/batchnorm/RsqrtRsqrt<transformer_block_9/layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2<
:transformer_block_9/layer_normalization_19/batchnorm/Rsqrt†
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_9_layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02I
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp√
8transformer_block_9/layer_normalization_19/batchnorm/mulMul>transformer_block_9/layer_normalization_19/batchnorm/Rsqrt:y:0Otransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2:
8transformer_block_9/layer_normalization_19/batchnorm/mulУ
:transformer_block_9/layer_normalization_19/batchnorm/mul_1Multransformer_block_9/add_1:z:0<transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_19/batchnorm/mul_1ґ
:transformer_block_9/layer_normalization_19/batchnorm/mul_2Mul@transformer_block_9/layer_normalization_19/moments/mean:output:0<transformer_block_9/layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_19/batchnorm/mul_2Ф
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_9_layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02E
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpњ
8transformer_block_9/layer_normalization_19/batchnorm/subSubKtransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp:value:0>transformer_block_9/layer_normalization_19/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2:
8transformer_block_9/layer_normalization_19/batchnorm/subґ
:transformer_block_9/layer_normalization_19/batchnorm/add_1AddV2>transformer_block_9/layer_normalization_19/batchnorm/mul_1:z:0<transformer_block_9/layer_normalization_19/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2<
:transformer_block_9/layer_normalization_19/batchnorm/add_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_2/ConstЊ
flatten_2/ReshapeReshape>transformer_block_9/layer_normalization_19/batchnorm/add_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_2/Reshapex
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisЊ
concatenate_2/concatConcatV2flatten_2/Reshape:output:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€И2
concatenate_2/concat©
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	И@*
dtype02 
dense_26/MatMul/ReadVariableOp•
dense_26/MatMulMatMulconcatenate_2/concat:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/MatMulІ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_26/BiasAdd/ReadVariableOp•
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/Reluy
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_24/dropout/Const©
dropout_24/dropout/MulMuldense_26/Relu:activations:0!dropout_24/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_24/dropout/Mul
dropout_24/dropout/ShapeShapedense_26/Relu:activations:0*
T0*
_output_shapes
:2
dropout_24/dropout/Shapeо
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*

seed*
seed221
/dropout_24/dropout/random_uniform/RandomUniformЛ
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_24/dropout/GreaterEqual/yк
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
dropout_24/dropout/GreaterEqual†
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_24/dropout/Cast¶
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_24/dropout/Mul_1®
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_27/MatMul/ReadVariableOp§
dense_27/MatMulMatMuldropout_24/dropout/Mul_1:z:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_27/MatMulІ
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_27/BiasAdd/ReadVariableOp•
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_27/Reluy
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_25/dropout/Const©
dropout_25/dropout/MulMuldense_27/Relu:activations:0!dropout_25/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_25/dropout/Mul
dropout_25/dropout/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
dropout_25/dropout/Shapeо
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*

seed*
seed221
/dropout_25/dropout/random_uniform/RandomUniformЛ
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_25/dropout/GreaterEqual/yк
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
dropout_25/dropout/GreaterEqual†
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_25/dropout/Cast¶
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_25/dropout/Mul_1®
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_28/MatMul/ReadVariableOp§
dense_28/MatMulMatMuldropout_25/dropout/Mul_1:z:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_28/MatMulІ
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_28/BiasAdd/ReadVariableOp•
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_28/BiasAddЬ
IdentityIdentitydense_28/BiasAdd:output:0:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp<^token_and_position_embedding_4/embedding_8/embedding_lookup<^token_and_position_embedding_4/embedding_9/embedding_lookupD^transformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpD^transformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpH^transformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpO^transformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpY^transformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_9/multi_head_attention_9/key/add/ReadVariableOpL^transformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpD^transformer_block_9/multi_head_attention_9/query/add/ReadVariableOpN^transformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpD^transformer_block_9/multi_head_attention_9/value/add/ReadVariableOpN^transformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpA^transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpA^transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOpC^transformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ђ
_input_shapesЪ
Ч:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2z
;token_and_position_embedding_4/embedding_8/embedding_lookup;token_and_position_embedding_4/embedding_8/embedding_lookup2z
;token_and_position_embedding_4/embedding_9/embedding_lookup;token_and_position_embedding_4/embedding_9/embedding_lookup2К
Ctransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_18/batchnorm/ReadVariableOp2Т
Gtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_18/batchnorm/mul/ReadVariableOp2К
Ctransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOpCtransformer_block_9/layer_normalization_19/batchnorm/ReadVariableOp2Т
Gtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOpGtransformer_block_9/layer_normalization_19/batchnorm/mul/ReadVariableOp2†
Ntransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOpNtransformer_block_9/multi_head_attention_9/attention_output/add/ReadVariableOp2і
Xtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_9/multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2Ж
Atransformer_block_9/multi_head_attention_9/key/add/ReadVariableOpAtransformer_block_9/multi_head_attention_9/key/add/ReadVariableOp2Ъ
Ktransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOpKtransformer_block_9/multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_9/multi_head_attention_9/query/add/ReadVariableOpCtransformer_block_9/multi_head_attention_9/query/add/ReadVariableOp2Ю
Mtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOpMtransformer_block_9/multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_9/multi_head_attention_9/value/add/ReadVariableOpCtransformer_block_9/multi_head_attention_9/value/add/ReadVariableOp2Ю
Mtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOpMtransformer_block_9/multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2Д
@transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_24/BiasAdd/ReadVariableOp2И
Btransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_24/Tensordot/ReadVariableOp2Д
@transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp@transformer_block_9/sequential_9/dense_25/BiasAdd/ReadVariableOp2И
Btransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOpBtransformer_block_9/sequential_9/dense_25/Tensordot/ReadVariableOp:R N
(
_output_shapes
:€€€€€€€€€ДR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
ХА
б
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_43275

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
7sequential_9_dense_24_tensordot_readvariableop_resource9
5sequential_9_dense_24_biasadd_readvariableop_resource;
7sequential_9_dense_25_tensordot_readvariableop_resource9
5sequential_9_dense_25_biasadd_readvariableop_resource@
<layer_normalization_19_batchnorm_mul_readvariableop_resource<
8layer_normalization_19_batchnorm_readvariableop_resource
identityИҐ/layer_normalization_18/batchnorm/ReadVariableOpҐ3layer_normalization_18/batchnorm/mul/ReadVariableOpҐ/layer_normalization_19/batchnorm/ReadVariableOpҐ3layer_normalization_19/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_9/attention_output/add/ReadVariableOpҐDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_9/key/add/ReadVariableOpҐ7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_9/query/add/ReadVariableOpҐ9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_9/value/add/ReadVariableOpҐ9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpҐ,sequential_9/dense_24/BiasAdd/ReadVariableOpҐ.sequential_9/dense_24/Tensordot/ReadVariableOpҐ,sequential_9/dense_25/BiasAdd/ReadVariableOpҐ.sequential_9/dense_25/Tensordot/ReadVariableOp€
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02;
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOpО
*multi_head_attention_9/query/einsum/EinsumEinsuminputsAmulti_head_attention_9/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2,
*multi_head_attention_9/query/einsum/Einsum№
/multi_head_attention_9/query/add/ReadVariableOpReadVariableOp8multi_head_attention_9_query_add_readvariableop_resource*
_output_shapes
:	А*
dtype021
/multi_head_attention_9/query/add/ReadVariableOpц
 multi_head_attention_9/query/addAddV23multi_head_attention_9/query/einsum/Einsum:output:07multi_head_attention_9/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 multi_head_attention_9/query/addщ
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_9_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype029
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOpИ
(multi_head_attention_9/key/einsum/EinsumEinsuminputs?multi_head_attention_9/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2*
(multi_head_attention_9/key/einsum/Einsum÷
-multi_head_attention_9/key/add/ReadVariableOpReadVariableOp6multi_head_attention_9_key_add_readvariableop_resource*
_output_shapes
:	А*
dtype02/
-multi_head_attention_9/key/add/ReadVariableOpо
multi_head_attention_9/key/addAddV21multi_head_attention_9/key/einsum/Einsum:output:05multi_head_attention_9/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
multi_head_attention_9/key/add€
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_9_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02;
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOpО
*multi_head_attention_9/value/einsum/EinsumEinsuminputsAmulti_head_attention_9/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationabc,cde->abde2,
*multi_head_attention_9/value/einsum/Einsum№
/multi_head_attention_9/value/add/ReadVariableOpReadVariableOp8multi_head_attention_9_value_add_readvariableop_resource*
_output_shapes
:	А*
dtype021
/multi_head_attention_9/value/add/ReadVariableOpц
 multi_head_attention_9/value/addAddV23multi_head_attention_9/value/einsum/Einsum:output:07multi_head_attention_9/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2"
 multi_head_attention_9/value/addБ
multi_head_attention_9/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *уµ=2
multi_head_attention_9/Mul/y«
multi_head_attention_9/MulMul$multi_head_attention_9/query/add:z:0%multi_head_attention_9/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
multi_head_attention_9/Mulь
$multi_head_attention_9/einsum/EinsumEinsum"multi_head_attention_9/key/add:z:0multi_head_attention_9/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€*
equationaecd,abcd->acbe2&
$multi_head_attention_9/einsum/Einsumƒ
&multi_head_attention_9/softmax/SoftmaxSoftmax-multi_head_attention_9/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€2(
&multi_head_attention_9/softmax/Softmax°
,multi_head_attention_9/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,multi_head_attention_9/dropout/dropout/ConstВ
*multi_head_attention_9/dropout/dropout/MulMul0multi_head_attention_9/softmax/Softmax:softmax:05multi_head_attention_9/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€2,
*multi_head_attention_9/dropout/dropout/MulЉ
,multi_head_attention_9/dropout/dropout/ShapeShape0multi_head_attention_9/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_9/dropout/dropout/Shape•
Cmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_9/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
dtype0*

seed2E
Cmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_9/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_9/dropout/dropout/GreaterEqual/y¬
3multi_head_attention_9/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_9/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_9/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€25
3multi_head_attention_9/dropout/dropout/GreaterEqualд
+multi_head_attention_9/dropout/dropout/CastCast7multi_head_attention_9/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€2-
+multi_head_attention_9/dropout/dropout/Castю
,multi_head_attention_9/dropout/dropout/Mul_1Mul.multi_head_attention_9/dropout/dropout/Mul:z:0/multi_head_attention_9/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€2.
,multi_head_attention_9/dropout/dropout/Mul_1Х
&multi_head_attention_9/einsum_1/EinsumEinsum0multi_head_attention_9/dropout/dropout/Mul_1:z:0$multi_head_attention_9/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€А*
equationacbe,aecd->abcd2(
&multi_head_attention_9/einsum_1/Einsum†
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_9_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:АА*
dtype02F
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp‘
5multi_head_attention_9/attention_output/einsum/EinsumEinsum/multi_head_attention_9/einsum_1/Einsum:output:0Lmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€А*
equationabcd,cde->abe27
5multi_head_attention_9/attention_output/einsum/Einsumщ
:multi_head_attention_9/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_9_attention_output_add_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:multi_head_attention_9/attention_output/add/ReadVariableOpЮ
+multi_head_attention_9/attention_output/addAddV2>multi_head_attention_9/attention_output/einsum/Einsum:output:0Bmulti_head_attention_9/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2-
+multi_head_attention_9/attention_output/addy
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_22/dropout/Const¬
dropout_22/dropout/MulMul/multi_head_attention_9/attention_output/add:z:0!dropout_22/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_22/dropout/MulУ
dropout_22/dropout/ShapeShape/multi_head_attention_9/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shapeу
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А*
dtype0*

seed*
seed221
/dropout_22/dropout/random_uniform/RandomUniformЛ
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_22/dropout/GreaterEqual/yп
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
dropout_22/dropout/GreaterEqual•
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€А2
dropout_22/dropout/CastЂ
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_22/dropout/Mul_1p
addAddV2inputsdropout_22/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2
addЄ
5layer_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_18/moments/mean/reduction_indicesв
#layer_normalization_18/moments/meanMeanadd:z:0>layer_normalization_18/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2%
#layer_normalization_18/moments/meanќ
+layer_normalization_18/moments/StopGradientStopGradient,layer_normalization_18/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2-
+layer_normalization_18/moments/StopGradientп
0layer_normalization_18/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_18/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А22
0layer_normalization_18/moments/SquaredDifferenceј
9layer_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_18/moments/variance/reduction_indicesЫ
'layer_normalization_18/moments/varianceMean4layer_normalization_18/moments/SquaredDifference:z:0Blayer_normalization_18/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2)
'layer_normalization_18/moments/varianceХ
&layer_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_18/batchnorm/add/yо
$layer_normalization_18/batchnorm/addAddV20layer_normalization_18/moments/variance:output:0/layer_normalization_18/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2&
$layer_normalization_18/batchnorm/addє
&layer_normalization_18/batchnorm/RsqrtRsqrt(layer_normalization_18/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2(
&layer_normalization_18/batchnorm/Rsqrtд
3layer_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3layer_normalization_18/batchnorm/mul/ReadVariableOpу
$layer_normalization_18/batchnorm/mulMul*layer_normalization_18/batchnorm/Rsqrt:y:0;layer_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_18/batchnorm/mulЅ
&layer_normalization_18/batchnorm/mul_1Muladd:z:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/mul_1ж
&layer_normalization_18/batchnorm/mul_2Mul,layer_normalization_18/moments/mean:output:0(layer_normalization_18/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/mul_2Ў
/layer_normalization_18/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/layer_normalization_18/batchnorm/ReadVariableOpп
$layer_normalization_18/batchnorm/subSub7layer_normalization_18/batchnorm/ReadVariableOp:value:0*layer_normalization_18/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_18/batchnorm/subж
&layer_normalization_18/batchnorm/add_1AddV2*layer_normalization_18/batchnorm/mul_1:z:0(layer_normalization_18/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_18/batchnorm/add_1Џ
.sequential_9/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_24_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype020
.sequential_9/dense_24/Tensordot/ReadVariableOpЦ
$sequential_9/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_24/Tensordot/axesЭ
$sequential_9/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_24/Tensordot/free®
%sequential_9/dense_24/Tensordot/ShapeShape*layer_normalization_18/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_9/dense_24/Tensordot/Shape†
-sequential_9/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_24/Tensordot/GatherV2/axisњ
(sequential_9/dense_24/Tensordot/GatherV2GatherV2.sequential_9/dense_24/Tensordot/Shape:output:0-sequential_9/dense_24/Tensordot/free:output:06sequential_9/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_24/Tensordot/GatherV2§
/sequential_9/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_24/Tensordot/GatherV2_1/axis≈
*sequential_9/dense_24/Tensordot/GatherV2_1GatherV2.sequential_9/dense_24/Tensordot/Shape:output:0-sequential_9/dense_24/Tensordot/axes:output:08sequential_9/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_24/Tensordot/GatherV2_1Ш
%sequential_9/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_24/Tensordot/ConstЎ
$sequential_9/dense_24/Tensordot/ProdProd1sequential_9/dense_24/Tensordot/GatherV2:output:0.sequential_9/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_24/Tensordot/ProdЬ
'sequential_9/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_24/Tensordot/Const_1а
&sequential_9/dense_24/Tensordot/Prod_1Prod3sequential_9/dense_24/Tensordot/GatherV2_1:output:00sequential_9/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_24/Tensordot/Prod_1Ь
+sequential_9/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_24/Tensordot/concat/axisЮ
&sequential_9/dense_24/Tensordot/concatConcatV2-sequential_9/dense_24/Tensordot/free:output:0-sequential_9/dense_24/Tensordot/axes:output:04sequential_9/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_24/Tensordot/concatд
%sequential_9/dense_24/Tensordot/stackPack-sequential_9/dense_24/Tensordot/Prod:output:0/sequential_9/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_24/Tensordot/stackч
)sequential_9/dense_24/Tensordot/transpose	Transpose*layer_normalization_18/batchnorm/add_1:z:0/sequential_9/dense_24/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2+
)sequential_9/dense_24/Tensordot/transposeч
'sequential_9/dense_24/Tensordot/ReshapeReshape-sequential_9/dense_24/Tensordot/transpose:y:0.sequential_9/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_9/dense_24/Tensordot/Reshapeч
&sequential_9/dense_24/Tensordot/MatMulMatMul0sequential_9/dense_24/Tensordot/Reshape:output:06sequential_9/dense_24/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_9/dense_24/Tensordot/MatMulЭ
'sequential_9/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2)
'sequential_9/dense_24/Tensordot/Const_2†
-sequential_9/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_24/Tensordot/concat_1/axisЂ
(sequential_9/dense_24/Tensordot/concat_1ConcatV21sequential_9/dense_24/Tensordot/GatherV2:output:00sequential_9/dense_24/Tensordot/Const_2:output:06sequential_9/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_24/Tensordot/concat_1й
sequential_9/dense_24/TensordotReshape0sequential_9/dense_24/Tensordot/MatMul:product:01sequential_9/dense_24/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
sequential_9/dense_24/Tensordotѕ
,sequential_9/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_24/BiasAdd/ReadVariableOpа
sequential_9/dense_24/BiasAddBiasAdd(sequential_9/dense_24/Tensordot:output:04sequential_9/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_24/BiasAddЯ
sequential_9/dense_24/ReluRelu&sequential_9/dense_24/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_24/ReluЏ
.sequential_9/dense_25/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_25_tensordot_readvariableop_resource* 
_output_shapes
:
АА*
dtype020
.sequential_9/dense_25/Tensordot/ReadVariableOpЦ
$sequential_9/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_9/dense_25/Tensordot/axesЭ
$sequential_9/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_9/dense_25/Tensordot/free¶
%sequential_9/dense_25/Tensordot/ShapeShape(sequential_9/dense_24/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_9/dense_25/Tensordot/Shape†
-sequential_9/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_25/Tensordot/GatherV2/axisњ
(sequential_9/dense_25/Tensordot/GatherV2GatherV2.sequential_9/dense_25/Tensordot/Shape:output:0-sequential_9/dense_25/Tensordot/free:output:06sequential_9/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_9/dense_25/Tensordot/GatherV2§
/sequential_9/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_9/dense_25/Tensordot/GatherV2_1/axis≈
*sequential_9/dense_25/Tensordot/GatherV2_1GatherV2.sequential_9/dense_25/Tensordot/Shape:output:0-sequential_9/dense_25/Tensordot/axes:output:08sequential_9/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_9/dense_25/Tensordot/GatherV2_1Ш
%sequential_9/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_9/dense_25/Tensordot/ConstЎ
$sequential_9/dense_25/Tensordot/ProdProd1sequential_9/dense_25/Tensordot/GatherV2:output:0.sequential_9/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_9/dense_25/Tensordot/ProdЬ
'sequential_9/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_9/dense_25/Tensordot/Const_1а
&sequential_9/dense_25/Tensordot/Prod_1Prod3sequential_9/dense_25/Tensordot/GatherV2_1:output:00sequential_9/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_9/dense_25/Tensordot/Prod_1Ь
+sequential_9/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_9/dense_25/Tensordot/concat/axisЮ
&sequential_9/dense_25/Tensordot/concatConcatV2-sequential_9/dense_25/Tensordot/free:output:0-sequential_9/dense_25/Tensordot/axes:output:04sequential_9/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_9/dense_25/Tensordot/concatд
%sequential_9/dense_25/Tensordot/stackPack-sequential_9/dense_25/Tensordot/Prod:output:0/sequential_9/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_25/Tensordot/stackх
)sequential_9/dense_25/Tensordot/transpose	Transpose(sequential_9/dense_24/Relu:activations:0/sequential_9/dense_25/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2+
)sequential_9/dense_25/Tensordot/transposeч
'sequential_9/dense_25/Tensordot/ReshapeReshape-sequential_9/dense_25/Tensordot/transpose:y:0.sequential_9/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_9/dense_25/Tensordot/Reshapeч
&sequential_9/dense_25/Tensordot/MatMulMatMul0sequential_9/dense_25/Tensordot/Reshape:output:06sequential_9/dense_25/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_9/dense_25/Tensordot/MatMulЭ
'sequential_9/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2)
'sequential_9/dense_25/Tensordot/Const_2†
-sequential_9/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/dense_25/Tensordot/concat_1/axisЂ
(sequential_9/dense_25/Tensordot/concat_1ConcatV21sequential_9/dense_25/Tensordot/GatherV2:output:00sequential_9/dense_25/Tensordot/Const_2:output:06sequential_9/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_9/dense_25/Tensordot/concat_1й
sequential_9/dense_25/TensordotReshape0sequential_9/dense_25/Tensordot/MatMul:product:01sequential_9/dense_25/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
sequential_9/dense_25/Tensordotѕ
,sequential_9/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_9/dense_25/BiasAdd/ReadVariableOpа
sequential_9/dense_25/BiasAddBiasAdd(sequential_9/dense_25/Tensordot:output:04sequential_9/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2
sequential_9/dense_25/BiasAddy
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_23/dropout/Constє
dropout_23/dropout/MulMul&sequential_9/dense_25/BiasAdd:output:0!dropout_23/dropout/Const:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_23/dropout/MulК
dropout_23/dropout/ShapeShape&sequential_9/dense_25/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shapeу
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А*
dtype0*

seed*
seed221
/dropout_23/dropout/random_uniform/RandomUniformЛ
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_23/dropout/GreaterEqual/yп
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€А2!
dropout_23/dropout/GreaterEqual•
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:€€€€€€€€€А2
dropout_23/dropout/CastЂ
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*,
_output_shapes
:€€€€€€€€€А2
dropout_23/dropout/Mul_1Ш
add_1AddV2*layer_normalization_18/batchnorm/add_1:z:0dropout_23/dropout/Mul_1:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2
add_1Є
5layer_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_19/moments/mean/reduction_indicesд
#layer_normalization_19/moments/meanMean	add_1:z:0>layer_normalization_19/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2%
#layer_normalization_19/moments/meanќ
+layer_normalization_19/moments/StopGradientStopGradient,layer_normalization_19/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€2-
+layer_normalization_19/moments/StopGradientс
0layer_normalization_19/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_19/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€А22
0layer_normalization_19/moments/SquaredDifferenceј
9layer_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_19/moments/variance/reduction_indicesЫ
'layer_normalization_19/moments/varianceMean4layer_normalization_19/moments/SquaredDifference:z:0Blayer_normalization_19/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
	keep_dims(2)
'layer_normalization_19/moments/varianceХ
&layer_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_19/batchnorm/add/yо
$layer_normalization_19/batchnorm/addAddV20layer_normalization_19/moments/variance:output:0/layer_normalization_19/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€2&
$layer_normalization_19/batchnorm/addє
&layer_normalization_19/batchnorm/RsqrtRsqrt(layer_normalization_19/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€2(
&layer_normalization_19/batchnorm/Rsqrtд
3layer_normalization_19/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_19_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3layer_normalization_19/batchnorm/mul/ReadVariableOpу
$layer_normalization_19/batchnorm/mulMul*layer_normalization_19/batchnorm/Rsqrt:y:0;layer_normalization_19/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_19/batchnorm/mul√
&layer_normalization_19/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/mul_1ж
&layer_normalization_19/batchnorm/mul_2Mul,layer_normalization_19/moments/mean:output:0(layer_normalization_19/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/mul_2Ў
/layer_normalization_19/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_19_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/layer_normalization_19/batchnorm/ReadVariableOpп
$layer_normalization_19/batchnorm/subSub7layer_normalization_19/batchnorm/ReadVariableOp:value:0*layer_normalization_19/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2&
$layer_normalization_19/batchnorm/subж
&layer_normalization_19/batchnorm/add_1AddV2*layer_normalization_19/batchnorm/mul_1:z:0(layer_normalization_19/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€А2(
&layer_normalization_19/batchnorm/add_1Ё
IdentityIdentity*layer_normalization_19/batchnorm/add_1:z:00^layer_normalization_18/batchnorm/ReadVariableOp4^layer_normalization_18/batchnorm/mul/ReadVariableOp0^layer_normalization_19/batchnorm/ReadVariableOp4^layer_normalization_19/batchnorm/mul/ReadVariableOp;^multi_head_attention_9/attention_output/add/ReadVariableOpE^multi_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_9/key/add/ReadVariableOp8^multi_head_attention_9/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/query/add/ReadVariableOp:^multi_head_attention_9/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_9/value/add/ReadVariableOp:^multi_head_attention_9/value/einsum/Einsum/ReadVariableOp-^sequential_9/dense_24/BiasAdd/ReadVariableOp/^sequential_9/dense_24/Tensordot/ReadVariableOp-^sequential_9/dense_25/BiasAdd/ReadVariableOp/^sequential_9/dense_25/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:€€€€€€€€€А::::::::::::::::2b
/layer_normalization_18/batchnorm/ReadVariableOp/layer_normalization_18/batchnorm/ReadVariableOp2j
3layer_normalization_18/batchnorm/mul/ReadVariableOp3layer_normalization_18/batchnorm/mul/ReadVariableOp2b
/layer_normalization_19/batchnorm/ReadVariableOp/layer_normalization_19/batchnorm/ReadVariableOp2j
3layer_normalization_19/batchnorm/mul/ReadVariableOp3layer_normalization_19/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_9/attention_output/add/ReadVariableOp:multi_head_attention_9/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_9/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_9/key/add/ReadVariableOp-multi_head_attention_9/key/add/ReadVariableOp2r
7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp7multi_head_attention_9/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/query/add/ReadVariableOp/multi_head_attention_9/query/add/ReadVariableOp2v
9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp9multi_head_attention_9/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_9/value/add/ReadVariableOp/multi_head_attention_9/value/add/ReadVariableOp2v
9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp9multi_head_attention_9/value/einsum/Einsum/ReadVariableOp2\
,sequential_9/dense_24/BiasAdd/ReadVariableOp,sequential_9/dense_24/BiasAdd/ReadVariableOp2`
.sequential_9/dense_24/Tensordot/ReadVariableOp.sequential_9/dense_24/Tensordot/ReadVariableOp2\
,sequential_9/dense_25/BiasAdd/ReadVariableOp,sequential_9/dense_25/BiasAdd/ReadVariableOp2`
.sequential_9/dense_25/Tensordot/ReadVariableOp.sequential_9/dense_25/Tensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
£
c
*__inference_dropout_24_layer_call_fn_45270

inputs
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_435802
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
«ј
Ж1
__inference__traced_save_45845
file_prefix:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	T
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
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableopO
Ksavev2_transformer_block_9_layer_normalization_18_gamma_read_readvariableopN
Jsavev2_transformer_block_9_layer_normalization_18_beta_read_readvariableopO
Ksavev2_transformer_block_9_layer_normalization_19_gamma_read_readvariableopN
Jsavev2_transformer_block_9_layer_normalization_19_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_4_embedding_8_embeddings_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_4_embedding_9_embeddings_m_read_readvariableop]
Ysavev2_adam_transformer_block_9_multi_head_attention_9_query_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_block_9_multi_head_attention_9_query_bias_m_read_readvariableop[
Wsavev2_adam_transformer_block_9_multi_head_attention_9_key_kernel_m_read_readvariableopY
Usavev2_adam_transformer_block_9_multi_head_attention_9_key_bias_m_read_readvariableop]
Ysavev2_adam_transformer_block_9_multi_head_attention_9_value_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_block_9_multi_head_attention_9_value_bias_m_read_readvariableoph
dsavev2_adam_transformer_block_9_multi_head_attention_9_attention_output_kernel_m_read_readvariableopf
bsavev2_adam_transformer_block_9_multi_head_attention_9_attention_output_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableopV
Rsavev2_adam_transformer_block_9_layer_normalization_18_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_9_layer_normalization_18_beta_m_read_readvariableopV
Rsavev2_adam_transformer_block_9_layer_normalization_19_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_9_layer_normalization_19_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_4_embedding_8_embeddings_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_4_embedding_9_embeddings_v_read_readvariableop]
Ysavev2_adam_transformer_block_9_multi_head_attention_9_query_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_block_9_multi_head_attention_9_query_bias_v_read_readvariableop[
Wsavev2_adam_transformer_block_9_multi_head_attention_9_key_kernel_v_read_readvariableopY
Usavev2_adam_transformer_block_9_multi_head_attention_9_key_bias_v_read_readvariableop]
Ysavev2_adam_transformer_block_9_multi_head_attention_9_value_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_block_9_multi_head_attention_9_value_bias_v_read_readvariableoph
dsavev2_adam_transformer_block_9_multi_head_attention_9_attention_output_kernel_v_read_readvariableopf
bsavev2_adam_transformer_block_9_multi_head_attention_9_attention_output_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableopV
Rsavev2_adam_transformer_block_9_layer_normalization_18_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_9_layer_normalization_18_beta_v_read_readvariableopV
Rsavev2_adam_transformer_block_9_layer_normalization_19_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_9_layer_normalization_19_beta_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameп/
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*Б/
valueч.Bф.XB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesї
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*≈
valueїBЄXB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices–/
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableopPsavev2_token_and_position_embedding_4_embedding_8_embeddings_read_readvariableopPsavev2_token_and_position_embedding_4_embedding_9_embeddings_read_readvariableopRsavev2_transformer_block_9_multi_head_attention_9_query_kernel_read_readvariableopPsavev2_transformer_block_9_multi_head_attention_9_query_bias_read_readvariableopPsavev2_transformer_block_9_multi_head_attention_9_key_kernel_read_readvariableopNsavev2_transformer_block_9_multi_head_attention_9_key_bias_read_readvariableopRsavev2_transformer_block_9_multi_head_attention_9_value_kernel_read_readvariableopPsavev2_transformer_block_9_multi_head_attention_9_value_bias_read_readvariableop]savev2_transformer_block_9_multi_head_attention_9_attention_output_kernel_read_readvariableop[savev2_transformer_block_9_multi_head_attention_9_attention_output_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableopKsavev2_transformer_block_9_layer_normalization_18_gamma_read_readvariableopJsavev2_transformer_block_9_layer_normalization_18_beta_read_readvariableopKsavev2_transformer_block_9_layer_normalization_19_gamma_read_readvariableopJsavev2_transformer_block_9_layer_normalization_19_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableopWsavev2_adam_token_and_position_embedding_4_embedding_8_embeddings_m_read_readvariableopWsavev2_adam_token_and_position_embedding_4_embedding_9_embeddings_m_read_readvariableopYsavev2_adam_transformer_block_9_multi_head_attention_9_query_kernel_m_read_readvariableopWsavev2_adam_transformer_block_9_multi_head_attention_9_query_bias_m_read_readvariableopWsavev2_adam_transformer_block_9_multi_head_attention_9_key_kernel_m_read_readvariableopUsavev2_adam_transformer_block_9_multi_head_attention_9_key_bias_m_read_readvariableopYsavev2_adam_transformer_block_9_multi_head_attention_9_value_kernel_m_read_readvariableopWsavev2_adam_transformer_block_9_multi_head_attention_9_value_bias_m_read_readvariableopdsavev2_adam_transformer_block_9_multi_head_attention_9_attention_output_kernel_m_read_readvariableopbsavev2_adam_transformer_block_9_multi_head_attention_9_attention_output_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableopRsavev2_adam_transformer_block_9_layer_normalization_18_gamma_m_read_readvariableopQsavev2_adam_transformer_block_9_layer_normalization_18_beta_m_read_readvariableopRsavev2_adam_transformer_block_9_layer_normalization_19_gamma_m_read_readvariableopQsavev2_adam_transformer_block_9_layer_normalization_19_beta_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableopWsavev2_adam_token_and_position_embedding_4_embedding_8_embeddings_v_read_readvariableopWsavev2_adam_token_and_position_embedding_4_embedding_9_embeddings_v_read_readvariableopYsavev2_adam_transformer_block_9_multi_head_attention_9_query_kernel_v_read_readvariableopWsavev2_adam_transformer_block_9_multi_head_attention_9_query_bias_v_read_readvariableopWsavev2_adam_transformer_block_9_multi_head_attention_9_key_kernel_v_read_readvariableopUsavev2_adam_transformer_block_9_multi_head_attention_9_key_bias_v_read_readvariableopYsavev2_adam_transformer_block_9_multi_head_attention_9_value_kernel_v_read_readvariableopWsavev2_adam_transformer_block_9_multi_head_attention_9_value_bias_v_read_readvariableopdsavev2_adam_transformer_block_9_multi_head_attention_9_attention_output_kernel_v_read_readvariableopbsavev2_adam_transformer_block_9_multi_head_attention_9_attention_output_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableopRsavev2_adam_transformer_block_9_layer_normalization_18_gamma_v_read_readvariableopQsavev2_adam_transformer_block_9_layer_normalization_18_beta_v_read_readvariableopRsavev2_adam_transformer_block_9_layer_normalization_19_gamma_v_read_readvariableopQsavev2_adam_transformer_block_9_layer_normalization_19_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *f
dtypes\
Z2X	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*µ
_input_shapes£
†: :А:А:А:А:	И@:@:@@:@:@:: : : : : :	А:
ДRА:АА:	А:АА:	А:АА:	А:АА:А:
АА:А:
АА:А:А:А:А:А: : :А:А:	И@:@:@@:@:@::	А:
ДRА:АА:	А:АА:	А:АА:	А:АА:А:
АА:А:
АА:А:А:А:А:А:А:А:	И@:@:@@:@:@::	А:
ДRА:АА:	А:АА:	А:АА:	А:АА:А:
АА:А:
АА:А:А:А:А:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:%!

_output_shapes
:	И@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	А:&"
 
_output_shapes
:
ДRА:*&
$
_output_shapes
:АА:%!

_output_shapes
:	А:*&
$
_output_shapes
:АА:%!

_output_shapes
:	А:*&
$
_output_shapes
:АА:%!

_output_shapes
:	А:*&
$
_output_shapes
:АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:! 

_output_shapes	
:А:!!

_output_shapes	
:А:"

_output_shapes
: :#

_output_shapes
: :!$

_output_shapes	
:А:!%

_output_shapes	
:А:%&!

_output_shapes
:	И@: '

_output_shapes
:@:$( 

_output_shapes

:@@: )

_output_shapes
:@:$* 

_output_shapes

:@: +

_output_shapes
::%,!

_output_shapes
:	А:&-"
 
_output_shapes
:
ДRА:*.&
$
_output_shapes
:АА:%/!

_output_shapes
:	А:*0&
$
_output_shapes
:АА:%1!

_output_shapes
:	А:*2&
$
_output_shapes
:АА:%3!

_output_shapes
:	А:*4&
$
_output_shapes
:АА:!5

_output_shapes	
:А:&6"
 
_output_shapes
:
АА:!7

_output_shapes	
:А:&8"
 
_output_shapes
:
АА:!9

_output_shapes	
:А:!:

_output_shapes	
:А:!;

_output_shapes	
:А:!<

_output_shapes	
:А:!=

_output_shapes	
:А:!>

_output_shapes	
:А:!?

_output_shapes	
:А:%@!

_output_shapes
:	И@: A

_output_shapes
:@:$B 

_output_shapes

:@@: C

_output_shapes
:@:$D 

_output_shapes

:@: E

_output_shapes
::%F!

_output_shapes
:	А:&G"
 
_output_shapes
:
ДRА:*H&
$
_output_shapes
:АА:%I!

_output_shapes
:	А:*J&
$
_output_shapes
:АА:%K!

_output_shapes
:	А:*L&
$
_output_shapes
:АА:%M!

_output_shapes
:	А:*N&
$
_output_shapes
:АА:!O

_output_shapes	
:А:&P"
 
_output_shapes
:
АА:!Q

_output_shapes	
:А:&R"
 
_output_shapes
:
АА:!S

_output_shapes	
:А:!T

_output_shapes	
:А:!U

_output_shapes	
:А:!V

_output_shapes	
:А:!W

_output_shapes	
:А:X

_output_shapes
: 
“
І
,__inference_sequential_9_layer_call_fn_42963
dense_24_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCalldense_24_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_429522
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:€€€€€€€€€А
(
_user_specified_namedense_24_input"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*л
serving_default„
=
input_101
serving_default_input_10:0€€€€€€€€€
<
input_91
serving_default_input_9:0€€€€€€€€€ДR<
dense_280
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Цж
р.
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
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
Б__call__
+В&call_and_return_all_conditional_losses
Г_default_save_signature"Џ*
_tf_keras_networkЊ*{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_4", "inbound_nodes": [[["input_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["token_and_position_embedding_4", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [5000]}, "pool_size": {"class_name": "__tuple__", "items": [5000]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_9", "inbound_nodes": [[["average_pooling1d_4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["transformer_block_9", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["flatten_2", 0, 0, {}], ["input_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_24", "inbound_nodes": [[["dense_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["dropout_24", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_25", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["dropout_25", 0, 0, {}]]]}], "input_layers": [["input_9", 0, 0], ["input_10", 0, 0]], "output_layers": [["dense_28", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
с"о
_tf_keras_input_layerќ{"class_name": "InputLayer", "name": "input_9", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}}
з
	token_emb
pos_emb
trainable_variables
regularization_losses
	variables
	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"Ї
_tf_keras_layer†{"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
љ	
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
 regularization_losses
!	variables
"	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"з
_tf_keras_layerЌ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500, 128]}}
Н
#trainable_variables
$regularization_losses
%	variables
&	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"ь
_tf_keras_layerв{"class_name": "AveragePooling1D", "name": "average_pooling1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [5000]}, "pool_size": {"class_name": "__tuple__", "items": [5000]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Д
'att
(ffn
)
layernorm1
*
layernorm2
+dropout1
,dropout2
-trainable_variables
.regularization_losses
/	variables
0	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"•
_tf_keras_layerЛ{"class_name": "TransformerBlock", "name": "transformer_block_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
и
1trainable_variables
2regularization_losses
3	variables
4	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"„
_tf_keras_layerљ{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
л"и
_tf_keras_input_layer»{"class_name": "InputLayer", "name": "input_10", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}}
ѕ
5trainable_variables
6regularization_losses
7	variables
8	keras_api
О__call__
+П&call_and_return_all_conditional_losses"Њ
_tf_keras_layer§{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 256]}, {"class_name": "TensorShape", "items": [null, 8]}]}
ц

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 264}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 264]}}
й
?trainable_variables
@regularization_losses
A	variables
B	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ф

Ckernel
Dbias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"Ќ
_tf_keras_layer≥{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
й
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
х

Mkernel
Nbias
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "Dense", "name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
џ

Sbeta_1

Tbeta_2
	Udecay
Vlearning_rate
WitermЌmќ9mѕ:m–Cm—Dm“Mm”Nm‘Xm’Ym÷Zm„[mЎ\mў]mЏ^mџ_m№`mЁamёbmяcmаdmбemвfmгgmдhmеimжvзvи9vй:vкCvлDvмMvнNvоXvпYvрZvс[vт\vу]vф^vх_vц`vчavшbvщcvъdvыevьfvэgvюhv€ivА"
	optimizer
ж
X0
Y1
2
3
Z4
[5
\6
]7
^8
_9
`10
a11
b12
c13
d14
e15
f16
g17
h18
i19
920
:21
C22
D23
M24
N25"
trackable_list_wrapper
 "
trackable_list_wrapper
ц
X0
Y1
2
3
4
5
Z6
[7
\8
]9
^10
_11
`12
a13
b14
c15
d16
e17
f18
g19
h20
i21
922
:23
C24
D25
M26
N27"
trackable_list_wrapper
ќ
jnon_trainable_variables
trainable_variables

klayers
regularization_losses
	variables
llayer_metrics
mmetrics
nlayer_regularization_losses
Б__call__
Г_default_save_signature
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
-
Ъserving_default"
signature_map
±
X
embeddings
otrainable_variables
pregularization_losses
q	variables
r	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"Р
_tf_keras_layerц{"class_name": "Embedding", "name": "embedding_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 5, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}}
Ѓ
Y
embeddings
strainable_variables
tregularization_losses
u	variables
v	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"Н
_tf_keras_layerу{"class_name": "Embedding", "name": "embedding_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10500, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
∞
wnon_trainable_variables
trainable_variables

xlayers
regularization_losses
ylayer_metrics
	variables
zlayer_regularization_losses
{metrics
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_4/gamma
):'А2batch_normalization_4/beta
2:0А (2!batch_normalization_4/moving_mean
6:4А (2%batch_normalization_4/moving_variance
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
±
|non_trainable_variables
trainable_variables

}layers
 regularization_losses
~layer_metrics
!	variables
layer_regularization_losses
Аmetrics
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Бnon_trainable_variables
#trainable_variables
Вlayers
$regularization_losses
Гlayer_metrics
%	variables
 Дlayer_regularization_losses
Еmetrics
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
К
Ж_query_dense
З
_key_dense
И_value_dense
Й_softmax
К_dropout_layer
Л_output_dense
Мtrainable_variables
Нregularization_losses
О	variables
П	keras_api
Я__call__
+†&call_and_return_all_conditional_losses"Ж
_tf_keras_layerм{"class_name": "MultiHeadAttention", "name": "multi_head_attention_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_9", "trainable": true, "dtype": "float32", "num_heads": 2, "key_dim": 128, "value_dim": 128, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
і
Рlayer_with_weights-0
Рlayer-0
Сlayer_with_weights-1
Сlayer-1
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses"Ќ
_tf_keras_sequentialЃ{"class_name": "Sequential", "name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_24_input"}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_24_input"}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
к
	Цaxis
	fgamma
gbeta
Чtrainable_variables
Шregularization_losses
Щ	variables
Ъ	keras_api
£__call__
+§&call_and_return_all_conditional_losses"µ
_tf_keras_layerЫ{"class_name": "LayerNormalization", "name": "layer_normalization_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_18", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 128]}}
к
	Ыaxis
	hgamma
ibeta
Ьtrainable_variables
Эregularization_losses
Ю	variables
Я	keras_api
•__call__
+¶&call_and_return_all_conditional_losses"µ
_tf_keras_layerЫ{"class_name": "LayerNormalization", "name": "layer_normalization_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_19", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 128]}}
н
†trainable_variables
°regularization_losses
Ґ	variables
£	keras_api
І__call__
+®&call_and_return_all_conditional_losses"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
н
§trainable_variables
•regularization_losses
¶	variables
І	keras_api
©__call__
+™&call_and_return_all_conditional_losses"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
Ц
Z0
[1
\2
]3
^4
_5
`6
a7
b8
c9
d10
e11
f12
g13
h14
i15"
trackable_list_wrapper
 "
trackable_list_wrapper
Ц
Z0
[1
\2
]3
^4
_5
`6
a7
b8
c9
d10
e11
f12
g13
h14
i15"
trackable_list_wrapper
µ
®non_trainable_variables
-trainable_variables
©layers
.regularization_losses
™layer_metrics
/	variables
 Ђlayer_regularization_losses
ђmetrics
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
≠non_trainable_variables
1trainable_variables
Ѓlayers
2regularization_losses
ѓlayer_metrics
3	variables
 ∞layer_regularization_losses
±metrics
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
≤non_trainable_variables
5trainable_variables
≥layers
6regularization_losses
іlayer_metrics
7	variables
 µlayer_regularization_losses
ґmetrics
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
": 	И@2dense_26/kernel
:@2dense_26/bias
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
µ
Јnon_trainable_variables
;trainable_variables
Єlayers
<regularization_losses
єlayer_metrics
=	variables
 Їlayer_regularization_losses
їmetrics
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Љnon_trainable_variables
?trainable_variables
љlayers
@regularization_losses
Њlayer_metrics
A	variables
 њlayer_regularization_losses
јmetrics
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_27/kernel
:@2dense_27/bias
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
µ
Ѕnon_trainable_variables
Etrainable_variables
¬layers
Fregularization_losses
√layer_metrics
G	variables
 ƒlayer_regularization_losses
≈metrics
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
∆non_trainable_variables
Itrainable_variables
«layers
Jregularization_losses
»layer_metrics
K	variables
 …layer_regularization_losses
 metrics
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_28/kernel
:2dense_28/bias
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
µ
Ћnon_trainable_variables
Otrainable_variables
ћlayers
Pregularization_losses
Ќlayer_metrics
Q	variables
 ќlayer_regularization_losses
ѕmetrics
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
H:F	А25token_and_position_embedding_4/embedding_8/embeddings
I:G
ДRА25token_and_position_embedding_4/embedding_9/embeddings
O:MАА27transformer_block_9/multi_head_attention_9/query/kernel
H:F	А25transformer_block_9/multi_head_attention_9/query/bias
M:KАА25transformer_block_9/multi_head_attention_9/key/kernel
F:D	А23transformer_block_9/multi_head_attention_9/key/bias
O:MАА27transformer_block_9/multi_head_attention_9/value/kernel
H:F	А25transformer_block_9/multi_head_attention_9/value/bias
Z:XАА2Btransformer_block_9/multi_head_attention_9/attention_output/kernel
O:MА2@transformer_block_9/multi_head_attention_9/attention_output/bias
#:!
АА2dense_24/kernel
:А2dense_24/bias
#:!
АА2dense_25/kernel
:А2dense_25/bias
?:=А20transformer_block_9/layer_normalization_18/gamma
>:<А2/transformer_block_9/layer_normalization_18/beta
?:=А20transformer_block_9/layer_normalization_19/gamma
>:<А2/transformer_block_9/layer_normalization_19/beta
.
0
1"
trackable_list_wrapper
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
12"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
–0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
X0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
X0"
trackable_list_wrapper
µ
—non_trainable_variables
otrainable_variables
“layers
pregularization_losses
”layer_metrics
q	variables
 ‘layer_regularization_losses
’metrics
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
'
Y0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
Y0"
trackable_list_wrapper
µ
÷non_trainable_variables
strainable_variables
„layers
tregularization_losses
Ўlayer_metrics
u	variables
 ўlayer_regularization_losses
Џmetrics
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
 
џpartial_output_shape
№full_output_shape

Zkernel
[bias
Ёtrainable_variables
ёregularization_losses
я	variables
а	keras_api
Ђ__call__
+ђ&call_and_return_all_conditional_losses"м
_tf_keras_layer“{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 2, 128], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 128]}}
∆
бpartial_output_shape
вfull_output_shape

\kernel
]bias
гtrainable_variables
дregularization_losses
е	variables
ж	keras_api
≠__call__
+Ѓ&call_and_return_all_conditional_losses"и
_tf_keras_layerќ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 2, 128], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 128]}}
 
зpartial_output_shape
иfull_output_shape

^kernel
_bias
йtrainable_variables
кregularization_losses
л	variables
м	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses"м
_tf_keras_layer“{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 2, 128], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 128]}}
л
нtrainable_variables
оregularization_losses
п	variables
р	keras_api
±__call__
+≤&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
з
сtrainable_variables
тregularization_losses
у	variables
ф	keras_api
≥__call__
+і&call_and_return_all_conditional_losses"“
_tf_keras_layerЄ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
я
хpartial_output_shape
цfull_output_shape

`kernel
abias
чtrainable_variables
шregularization_losses
щ	variables
ъ	keras_api
µ__call__
+ґ&call_and_return_all_conditional_losses"Б
_tf_keras_layerз{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 128], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 128]}}
X
Z0
[1
\2
]3
^4
_5
`6
a7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
Z0
[1
\2
]3
^4
_5
`6
a7"
trackable_list_wrapper
Є
ыnon_trainable_variables
Мtrainable_variables
ьlayers
Нregularization_losses
эlayer_metrics
О	variables
 юlayer_regularization_losses
€metrics
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
ю

bkernel
cbias
Аtrainable_variables
Бregularization_losses
В	variables
Г	keras_api
Ј__call__
+Є&call_and_return_all_conditional_losses"”
_tf_keras_layerє{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 128]}}
А

dkernel
ebias
Дtrainable_variables
Еregularization_losses
Ж	variables
З	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses"’
_tf_keras_layerї{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 128]}}
<
b0
c1
d2
e3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
b0
c1
d2
e3"
trackable_list_wrapper
Є
Иnon_trainable_variables
Тtrainable_variables
Йlayers
Уregularization_losses
Ф	variables
Кlayer_metrics
Лmetrics
 Мlayer_regularization_losses
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
Є
Нnon_trainable_variables
Чtrainable_variables
Оlayers
Шregularization_losses
Пlayer_metrics
Щ	variables
 Рlayer_regularization_losses
Сmetrics
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
Є
Тnon_trainable_variables
Ьtrainable_variables
Уlayers
Эregularization_losses
Фlayer_metrics
Ю	variables
 Хlayer_regularization_losses
Цmetrics
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Чnon_trainable_variables
†trainable_variables
Шlayers
°regularization_losses
Щlayer_metrics
Ґ	variables
 Ъlayer_regularization_losses
Ыmetrics
І__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ьnon_trainable_variables
§trainable_variables
Эlayers
•regularization_losses
Юlayer_metrics
¶	variables
 Яlayer_regularization_losses
†metrics
©__call__
+™&call_and_return_all_conditional_losses
'™"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
'0
(1
)2
*3
+4
,5"
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
њ

°total

Ґcount
£	variables
§	keras_api"Д
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
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
Є
•non_trainable_variables
Ёtrainable_variables
¶layers
ёregularization_losses
Іlayer_metrics
я	variables
 ®layer_regularization_losses
©metrics
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
Є
™non_trainable_variables
гtrainable_variables
Ђlayers
дregularization_losses
ђlayer_metrics
е	variables
 ≠layer_regularization_losses
Ѓmetrics
≠__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
Є
ѓnon_trainable_variables
йtrainable_variables
∞layers
кregularization_losses
±layer_metrics
л	variables
 ≤layer_regularization_losses
≥metrics
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
іnon_trainable_variables
нtrainable_variables
µlayers
оregularization_losses
ґlayer_metrics
п	variables
 Јlayer_regularization_losses
Єmetrics
±__call__
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
єnon_trainable_variables
сtrainable_variables
Їlayers
тregularization_losses
їlayer_metrics
у	variables
 Љlayer_regularization_losses
љmetrics
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
Є
Њnon_trainable_variables
чtrainable_variables
њlayers
шregularization_losses
јlayer_metrics
щ	variables
 Ѕlayer_regularization_losses
¬metrics
µ__call__
+ґ&call_and_return_all_conditional_losses
'ґ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
P
Ж0
З1
И2
Й3
К4
Л5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
Є
√non_trainable_variables
Аtrainable_variables
ƒlayers
Бregularization_losses
≈layer_metrics
В	variables
 ∆layer_regularization_losses
«metrics
Ј__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
Є
»non_trainable_variables
Дtrainable_variables
…layers
Еregularization_losses
 layer_metrics
Ж	variables
 Ћlayer_regularization_losses
ћmetrics
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Р0
С1"
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
:  (2total
:  (2count
0
°0
Ґ1"
trackable_list_wrapper
.
£	variables"
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
/:-А2"Adam/batch_normalization_4/gamma/m
.:,А2!Adam/batch_normalization_4/beta/m
':%	И@2Adam/dense_26/kernel/m
 :@2Adam/dense_26/bias/m
&:$@@2Adam/dense_27/kernel/m
 :@2Adam/dense_27/bias/m
&:$@2Adam/dense_28/kernel/m
 :2Adam/dense_28/bias/m
M:K	А2<Adam/token_and_position_embedding_4/embedding_8/embeddings/m
N:L
ДRА2<Adam/token_and_position_embedding_4/embedding_9/embeddings/m
T:RАА2>Adam/transformer_block_9/multi_head_attention_9/query/kernel/m
M:K	А2<Adam/transformer_block_9/multi_head_attention_9/query/bias/m
R:PАА2<Adam/transformer_block_9/multi_head_attention_9/key/kernel/m
K:I	А2:Adam/transformer_block_9/multi_head_attention_9/key/bias/m
T:RАА2>Adam/transformer_block_9/multi_head_attention_9/value/kernel/m
M:K	А2<Adam/transformer_block_9/multi_head_attention_9/value/bias/m
_:]АА2IAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/m
T:RА2GAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/m
(:&
АА2Adam/dense_24/kernel/m
!:А2Adam/dense_24/bias/m
(:&
АА2Adam/dense_25/kernel/m
!:А2Adam/dense_25/bias/m
D:BА27Adam/transformer_block_9/layer_normalization_18/gamma/m
C:AА26Adam/transformer_block_9/layer_normalization_18/beta/m
D:BА27Adam/transformer_block_9/layer_normalization_19/gamma/m
C:AА26Adam/transformer_block_9/layer_normalization_19/beta/m
/:-А2"Adam/batch_normalization_4/gamma/v
.:,А2!Adam/batch_normalization_4/beta/v
':%	И@2Adam/dense_26/kernel/v
 :@2Adam/dense_26/bias/v
&:$@@2Adam/dense_27/kernel/v
 :@2Adam/dense_27/bias/v
&:$@2Adam/dense_28/kernel/v
 :2Adam/dense_28/bias/v
M:K	А2<Adam/token_and_position_embedding_4/embedding_8/embeddings/v
N:L
ДRА2<Adam/token_and_position_embedding_4/embedding_9/embeddings/v
T:RАА2>Adam/transformer_block_9/multi_head_attention_9/query/kernel/v
M:K	А2<Adam/transformer_block_9/multi_head_attention_9/query/bias/v
R:PАА2<Adam/transformer_block_9/multi_head_attention_9/key/kernel/v
K:I	А2:Adam/transformer_block_9/multi_head_attention_9/key/bias/v
T:RАА2>Adam/transformer_block_9/multi_head_attention_9/value/kernel/v
M:K	А2<Adam/transformer_block_9/multi_head_attention_9/value/bias/v
_:]АА2IAdam/transformer_block_9/multi_head_attention_9/attention_output/kernel/v
T:RА2GAdam/transformer_block_9/multi_head_attention_9/attention_output/bias/v
(:&
АА2Adam/dense_24/kernel/v
!:А2Adam/dense_24/bias/v
(:&
АА2Adam/dense_25/kernel/v
!:А2Adam/dense_25/bias/v
D:BА27Adam/transformer_block_9/layer_normalization_18/gamma/v
C:AА26Adam/transformer_block_9/layer_normalization_18/beta/v
D:BА27Adam/transformer_block_9/layer_normalization_19/gamma/v
C:AА26Adam/transformer_block_9/layer_normalization_19/beta/v
к2з
'__inference_model_2_layer_call_fn_44596
'__inference_model_2_layer_call_fn_44023
'__inference_model_2_layer_call_fn_44658
'__inference_model_2_layer_call_fn_43889ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
B__inference_model_2_layer_call_and_return_conditional_losses_43682
B__inference_model_2_layer_call_and_return_conditional_losses_44340
B__inference_model_2_layer_call_and_return_conditional_losses_44534
B__inference_model_2_layer_call_and_return_conditional_losses_43754ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
И2Е
 __inference__wrapped_model_42668а
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *PҐM
KЪH
"К
input_9€€€€€€€€€ДR
"К
input_10€€€€€€€€€
г2а
>__inference_token_and_position_embedding_4_layer_call_fn_44691Э
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ю2ы
Y__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_44682Э
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ц2У
5__inference_batch_normalization_4_layer_call_fn_44842
5__inference_batch_normalization_4_layer_call_fn_44760
5__inference_batch_normalization_4_layer_call_fn_44855
5__inference_batch_normalization_4_layer_call_fn_44773і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
В2€
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44829
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44727
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44809
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44747і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
3__inference_average_pooling1d_4_layer_call_fn_42823”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
©2¶
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_42817”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
†2Э
3__inference_transformer_block_9_layer_call_fn_45204
3__inference_transformer_block_9_layer_call_fn_45167∞
І≤£
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_45130
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_45003∞
І≤£
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
”2–
)__inference_flatten_2_layer_call_fn_45215Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_flatten_2_layer_call_and_return_conditional_losses_45210Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_concatenate_2_layer_call_fn_45228Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_concatenate_2_layer_call_and_return_conditional_losses_45222Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_26_layer_call_fn_45248Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_26_layer_call_and_return_conditional_losses_45239Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
*__inference_dropout_24_layer_call_fn_45275
*__inference_dropout_24_layer_call_fn_45270і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
»2≈
E__inference_dropout_24_layer_call_and_return_conditional_losses_45265
E__inference_dropout_24_layer_call_and_return_conditional_losses_45260і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
(__inference_dense_27_layer_call_fn_45295Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_27_layer_call_and_return_conditional_losses_45286Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
*__inference_dropout_25_layer_call_fn_45317
*__inference_dropout_25_layer_call_fn_45322і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
»2≈
E__inference_dropout_25_layer_call_and_return_conditional_losses_45312
E__inference_dropout_25_layer_call_and_return_conditional_losses_45307і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
(__inference_dense_28_layer_call_fn_45341Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_28_layer_call_and_return_conditional_losses_45332Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“Bѕ
#__inference_signature_wrapper_44095input_10input_9"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
В2€ь
у≤п
FullArgSpece
args]ЪZ
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
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
В2€ь
у≤п
FullArgSpece
args]ЪZ
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
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
,__inference_sequential_9_layer_call_fn_45481
,__inference_sequential_9_layer_call_fn_42963
,__inference_sequential_9_layer_call_fn_42990
,__inference_sequential_9_layer_call_fn_45468ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
G__inference_sequential_9_layer_call_and_return_conditional_losses_42921
G__inference_sequential_9_layer_call_and_return_conditional_losses_45398
G__inference_sequential_9_layer_call_and_return_conditional_losses_45455
G__inference_sequential_9_layer_call_and_return_conditional_losses_42935ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_24_layer_call_fn_45521Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_24_layer_call_and_return_conditional_losses_45512Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_25_layer_call_fn_45560Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_25_layer_call_and_return_conditional_losses_45551Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ‘
 __inference__wrapped_model_42668ѓYXZ[\]^_`afgbcdehi9:CDMNZҐW
PҐM
KЪH
"К
input_9€€€€€€€€€ДR
"К
input_10€€€€€€€€€
™ "3™0
.
dense_28"К
dense_28€€€€€€€€€„
N__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_42817ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ѓ
3__inference_average_pooling1d_4_layer_call_fn_42823wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€“
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44727~AҐ>
7Ґ4
.К+
inputs€€€€€€€€€€€€€€€€€€А
p
™ "3Ґ0
)К&
0€€€€€€€€€€€€€€€€€€А
Ъ “
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44747~AҐ>
7Ґ4
.К+
inputs€€€€€€€€€€€€€€€€€€А
p 
™ "3Ґ0
)К&
0€€€€€€€€€€€€€€€€€€А
Ъ ¬
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44809n9Ґ6
/Ґ,
&К#
inputs€€€€€€€€€ДRА
p
™ "+Ґ(
!К
0€€€€€€€€€ДRА
Ъ ¬
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44829n9Ґ6
/Ґ,
&К#
inputs€€€€€€€€€ДRА
p 
™ "+Ґ(
!К
0€€€€€€€€€ДRА
Ъ ™
5__inference_batch_normalization_4_layer_call_fn_44760qAҐ>
7Ґ4
.К+
inputs€€€€€€€€€€€€€€€€€€А
p
™ "&К#€€€€€€€€€€€€€€€€€€А™
5__inference_batch_normalization_4_layer_call_fn_44773qAҐ>
7Ґ4
.К+
inputs€€€€€€€€€€€€€€€€€€А
p 
™ "&К#€€€€€€€€€€€€€€€€€€АЪ
5__inference_batch_normalization_4_layer_call_fn_44842a9Ґ6
/Ґ,
&К#
inputs€€€€€€€€€ДRА
p
™ "К€€€€€€€€€ДRАЪ
5__inference_batch_normalization_4_layer_call_fn_44855a9Ґ6
/Ґ,
&К#
inputs€€€€€€€€€ДRА
p 
™ "К€€€€€€€€€ДRА“
H__inference_concatenate_2_layer_call_and_return_conditional_losses_45222Е[ҐX
QҐN
LЪI
#К 
inputs/0€€€€€€€€€А
"К
inputs/1€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€И
Ъ ©
-__inference_concatenate_2_layer_call_fn_45228x[ҐX
QҐN
LЪI
#К 
inputs/0€€€€€€€€€А
"К
inputs/1€€€€€€€€€
™ "К€€€€€€€€€И≠
C__inference_dense_24_layer_call_and_return_conditional_losses_45512fbc4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€А
™ "*Ґ'
 К
0€€€€€€€€€А
Ъ Е
(__inference_dense_24_layer_call_fn_45521Ybc4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€А
™ "К€€€€€€€€€А≠
C__inference_dense_25_layer_call_and_return_conditional_losses_45551fde4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€А
™ "*Ґ'
 К
0€€€€€€€€€А
Ъ Е
(__inference_dense_25_layer_call_fn_45560Yde4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€А
™ "К€€€€€€€€€А§
C__inference_dense_26_layer_call_and_return_conditional_losses_45239]9:0Ґ-
&Ґ#
!К
inputs€€€€€€€€€И
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
(__inference_dense_26_layer_call_fn_45248P9:0Ґ-
&Ґ#
!К
inputs€€€€€€€€€И
™ "К€€€€€€€€€@£
C__inference_dense_27_layer_call_and_return_conditional_losses_45286\CD/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ {
(__inference_dense_27_layer_call_fn_45295OCD/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@£
C__inference_dense_28_layer_call_and_return_conditional_losses_45332\MN/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_28_layer_call_fn_45341OMN/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€•
E__inference_dropout_24_layer_call_and_return_conditional_losses_45260\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ •
E__inference_dropout_24_layer_call_and_return_conditional_losses_45265\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ }
*__inference_dropout_24_layer_call_fn_45270O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@}
*__inference_dropout_24_layer_call_fn_45275O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@•
E__inference_dropout_25_layer_call_and_return_conditional_losses_45307\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ •
E__inference_dropout_25_layer_call_and_return_conditional_losses_45312\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ }
*__inference_dropout_25_layer_call_fn_45317O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@}
*__inference_dropout_25_layer_call_fn_45322O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@¶
D__inference_flatten_2_layer_call_and_return_conditional_losses_45210^4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ~
)__inference_flatten_2_layer_call_fn_45215Q4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€А
™ "К€€€€€€€€€Ар
B__inference_model_2_layer_call_and_return_conditional_losses_43682©YXZ[\]^_`afgbcdehi9:CDMNbҐ_
XҐU
KЪH
"К
input_9€€€€€€€€€ДR
"К
input_10€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ р
B__inference_model_2_layer_call_and_return_conditional_losses_43754©YXZ[\]^_`afgbcdehi9:CDMNbҐ_
XҐU
KЪH
"К
input_9€€€€€€€€€ДR
"К
input_10€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ с
B__inference_model_2_layer_call_and_return_conditional_losses_44340™YXZ[\]^_`afgbcdehi9:CDMNcҐ`
YҐV
LЪI
#К 
inputs/0€€€€€€€€€ДR
"К
inputs/1€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ с
B__inference_model_2_layer_call_and_return_conditional_losses_44534™YXZ[\]^_`afgbcdehi9:CDMNcҐ`
YҐV
LЪI
#К 
inputs/0€€€€€€€€€ДR
"К
inputs/1€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ »
'__inference_model_2_layer_call_fn_43889ЬYXZ[\]^_`afgbcdehi9:CDMNbҐ_
XҐU
KЪH
"К
input_9€€€€€€€€€ДR
"К
input_10€€€€€€€€€
p

 
™ "К€€€€€€€€€»
'__inference_model_2_layer_call_fn_44023ЬYXZ[\]^_`afgbcdehi9:CDMNbҐ_
XҐU
KЪH
"К
input_9€€€€€€€€€ДR
"К
input_10€€€€€€€€€
p 

 
™ "К€€€€€€€€€…
'__inference_model_2_layer_call_fn_44596ЭYXZ[\]^_`afgbcdehi9:CDMNcҐ`
YҐV
LЪI
#К 
inputs/0€€€€€€€€€ДR
"К
inputs/1€€€€€€€€€
p

 
™ "К€€€€€€€€€…
'__inference_model_2_layer_call_fn_44658ЭYXZ[\]^_`afgbcdehi9:CDMNcҐ`
YҐV
LЪI
#К 
inputs/0€€€€€€€€€ДR
"К
inputs/1€€€€€€€€€
p 

 
™ "К€€€€€€€€€√
G__inference_sequential_9_layer_call_and_return_conditional_losses_42921xbcdeDҐA
:Ґ7
-К*
dense_24_input€€€€€€€€€А
p

 
™ "*Ґ'
 К
0€€€€€€€€€А
Ъ √
G__inference_sequential_9_layer_call_and_return_conditional_losses_42935xbcdeDҐA
:Ґ7
-К*
dense_24_input€€€€€€€€€А
p 

 
™ "*Ґ'
 К
0€€€€€€€€€А
Ъ ї
G__inference_sequential_9_layer_call_and_return_conditional_losses_45398pbcde<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€А
p

 
™ "*Ґ'
 К
0€€€€€€€€€А
Ъ ї
G__inference_sequential_9_layer_call_and_return_conditional_losses_45455pbcde<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€А
p 

 
™ "*Ґ'
 К
0€€€€€€€€€А
Ъ Ы
,__inference_sequential_9_layer_call_fn_42963kbcdeDҐA
:Ґ7
-К*
dense_24_input€€€€€€€€€А
p

 
™ "К€€€€€€€€€АЫ
,__inference_sequential_9_layer_call_fn_42990kbcdeDҐA
:Ґ7
-К*
dense_24_input€€€€€€€€€А
p 

 
™ "К€€€€€€€€€АУ
,__inference_sequential_9_layer_call_fn_45468cbcde<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€А
p

 
™ "К€€€€€€€€€АУ
,__inference_sequential_9_layer_call_fn_45481cbcde<Ґ9
2Ґ/
%К"
inputs€€€€€€€€€А
p 

 
™ "К€€€€€€€€€Ай
#__inference_signature_wrapper_44095ЅYXZ[\]^_`afgbcdehi9:CDMNlҐi
Ґ 
b™_
.
input_10"К
input_10€€€€€€€€€
-
input_9"К
input_9€€€€€€€€€ДR"3™0
.
dense_28"К
dense_28€€€€€€€€€ї
Y__inference_token_and_position_embedding_4_layer_call_and_return_conditional_losses_44682^YX+Ґ(
!Ґ
К
x€€€€€€€€€ДR
™ "+Ґ(
!К
0€€€€€€€€€ДRА
Ъ У
>__inference_token_and_position_embedding_4_layer_call_fn_44691QYX+Ґ(
!Ґ
К
x€€€€€€€€€ДR
™ "К€€€€€€€€€ДRА 
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_45003xZ[\]^_`afgbcdehi8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€А
p
™ "*Ґ'
 К
0€€€€€€€€€А
Ъ  
N__inference_transformer_block_9_layer_call_and_return_conditional_losses_45130xZ[\]^_`afgbcdehi8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€А
p 
™ "*Ґ'
 К
0€€€€€€€€€А
Ъ Ґ
3__inference_transformer_block_9_layer_call_fn_45167kZ[\]^_`afgbcdehi8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€АҐ
3__inference_transformer_block_9_layer_call_fn_45204kZ[\]^_`afgbcdehi8Ґ5
.Ґ+
%К"
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€А