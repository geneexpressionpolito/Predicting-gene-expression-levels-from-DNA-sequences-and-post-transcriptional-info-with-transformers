¬ђ%
њ!Х!
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
Ы
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
 И"serve*2.4.12v2.4.1-0-g85c8b2a817f8ЗМ
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:  *
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
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  * 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:	  *
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
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0
Ґ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
Ґ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и@* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	и@*
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
∆
5token_and_position_embedding_1/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_1/embedding_2/embeddings
њ
Itoken_and_position_embedding_1/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_1/embedding_2/embeddings*
_output_shapes

: *
dtype0
«
5token_and_position_embedding_1/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ДR *F
shared_name75token_and_position_embedding_1/embedding_3/embeddings
ј
Itoken_and_position_embedding_1/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_1/embedding_3/embeddings*
_output_shapes
:	ДR *
dtype0
ќ
7transformer_block_3/multi_head_attention_3/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_3/multi_head_attention_3/query/kernel
«
Ktransformer_block_3/multi_head_attention_3/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_3/multi_head_attention_3/query/kernel*"
_output_shapes
:  *
dtype0
∆
5transformer_block_3/multi_head_attention_3/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_3/multi_head_attention_3/query/bias
њ
Itransformer_block_3/multi_head_attention_3/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_3/multi_head_attention_3/query/bias*
_output_shapes

: *
dtype0
 
5transformer_block_3/multi_head_attention_3/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_3/multi_head_attention_3/key/kernel
√
Itransformer_block_3/multi_head_attention_3/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_3/multi_head_attention_3/key/kernel*"
_output_shapes
:  *
dtype0
¬
3transformer_block_3/multi_head_attention_3/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_3/multi_head_attention_3/key/bias
ї
Gtransformer_block_3/multi_head_attention_3/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_3/multi_head_attention_3/key/bias*
_output_shapes

: *
dtype0
ќ
7transformer_block_3/multi_head_attention_3/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_3/multi_head_attention_3/value/kernel
«
Ktransformer_block_3/multi_head_attention_3/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_3/multi_head_attention_3/value/kernel*"
_output_shapes
:  *
dtype0
∆
5transformer_block_3/multi_head_attention_3/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_3/multi_head_attention_3/value/bias
њ
Itransformer_block_3/multi_head_attention_3/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_3/multi_head_attention_3/value/bias*
_output_shapes

: *
dtype0
д
Btransformer_block_3/multi_head_attention_3/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_3/multi_head_attention_3/attention_output/kernel
Ё
Vtransformer_block_3/multi_head_attention_3/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_3/multi_head_attention_3/attention_output/kernel*"
_output_shapes
:  *
dtype0
Ў
@transformer_block_3/multi_head_attention_3/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_3/multi_head_attention_3/attention_output/bias
—
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
ґ
/transformer_block_3/layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_3/layer_normalization_6/gamma
ѓ
Ctransformer_block_3/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_3/layer_normalization_6/gamma*
_output_shapes
: *
dtype0
і
.transformer_block_3/layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_3/layer_normalization_6/beta
≠
Btransformer_block_3/layer_normalization_6/beta/Read/ReadVariableOpReadVariableOp.transformer_block_3/layer_normalization_6/beta*
_output_shapes
: *
dtype0
ґ
/transformer_block_3/layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_3/layer_normalization_7/gamma
ѓ
Ctransformer_block_3/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_3/layer_normalization_7/gamma*
_output_shapes
: *
dtype0
і
.transformer_block_3/layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_3/layer_normalization_7/beta
≠
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

NoOpNoOp
ЭУ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*„Т
valueћТB»Т BјТ
А
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

signatures
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
%
#_self_saveable_object_factories
У
	token_emb
pos_emb
#_self_saveable_object_factories
trainable_variables
 regularization_losses
!	variables
"	keras_api
Н

#kernel
$bias
#%_self_saveable_object_factories
&trainable_variables
'regularization_losses
(	variables
)	keras_api
w
#*_self_saveable_object_factories
+trainable_variables
,regularization_losses
-	variables
.	keras_api
Н

/kernel
0bias
#1_self_saveable_object_factories
2trainable_variables
3regularization_losses
4	variables
5	keras_api
w
#6_self_saveable_object_factories
7trainable_variables
8regularization_losses
9	variables
:	keras_api
w
#;_self_saveable_object_factories
<trainable_variables
=regularization_losses
>	variables
?	keras_api
Љ
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
#E_self_saveable_object_factories
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
Љ
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
#O_self_saveable_object_factories
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
w
#T_self_saveable_object_factories
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
≈
Yatt
Zffn
[
layernorm1
\
layernorm2
]dropout1
^dropout2
#__self_saveable_object_factories
`trainable_variables
aregularization_losses
b	variables
c	keras_api
w
#d_self_saveable_object_factories
etrainable_variables
fregularization_losses
g	variables
h	keras_api
%
#i_self_saveable_object_factories
w
#j_self_saveable_object_factories
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
Н

okernel
pbias
#q_self_saveable_object_factories
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
w
#v_self_saveable_object_factories
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
П

{kernel
|bias
#}_self_saveable_object_factories
~trainable_variables
regularization_losses
А	variables
Б	keras_api
|
$В_self_saveable_object_factories
Гtrainable_variables
Дregularization_losses
Е	variables
Ж	keras_api
Ф
Зkernel
	Иbias
$Й_self_saveable_object_factories
Кtrainable_variables
Лregularization_losses
М	variables
Н	keras_api
 
 
 
К
О0
П1
#2
$3
/4
05
A6
B7
K8
L9
Р10
С11
Т12
У13
Ф14
Х15
Ц16
Ч17
Ш18
Щ19
Ъ20
Ы21
Ь22
Э23
Ю24
Я25
o26
p27
{28
|29
З30
И31
 
™
О0
П1
#2
$3
/4
05
A6
B7
C8
D9
K10
L11
M12
N13
Р14
С15
Т16
У17
Ф18
Х19
Ц20
Ч21
Ш22
Щ23
Ъ24
Ы25
Ь26
Э27
Ю28
Я29
o30
p31
{32
|33
З34
И35
≤
†layer_metrics
trainable_variables
 °layer_regularization_losses
regularization_losses
	variables
Ґlayers
£non_trainable_variables
§metrics
 
Н
О
embeddings
$•_self_saveable_object_factories
¶trainable_variables
Іregularization_losses
®	variables
©	keras_api
Н
П
embeddings
$™_self_saveable_object_factories
Ђtrainable_variables
ђregularization_losses
≠	variables
Ѓ	keras_api
 

О0
П1
 

О0
П1
≤
ѓlayer_metrics
trainable_variables
 ∞layer_regularization_losses
 regularization_losses
!	variables
±layers
≤non_trainable_variables
≥metrics
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1
 

#0
$1
≤
іlayer_metrics
&trainable_variables
 µlayer_regularization_losses
'regularization_losses
(	variables
ґlayers
Јnon_trainable_variables
Єmetrics
 
 
 
 
≤
єlayer_metrics
+trainable_variables
 Їlayer_regularization_losses
,regularization_losses
-	variables
їlayers
Љnon_trainable_variables
љmetrics
[Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01
 

/0
01
≤
Њlayer_metrics
2trainable_variables
 њlayer_regularization_losses
3regularization_losses
4	variables
јlayers
Ѕnon_trainable_variables
¬metrics
 
 
 
 
≤
√layer_metrics
7trainable_variables
 ƒlayer_regularization_losses
8regularization_losses
9	variables
≈layers
∆non_trainable_variables
«metrics
 
 
 
 
≤
»layer_metrics
<trainable_variables
 …layer_regularization_losses
=regularization_losses
>	variables
 layers
Ћnon_trainable_variables
ћmetrics
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1
 

A0
B1
C2
D3
≤
Ќlayer_metrics
Ftrainable_variables
 ќlayer_regularization_losses
Gregularization_losses
H	variables
ѕlayers
–non_trainable_variables
—metrics
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1
 

K0
L1
M2
N3
≤
“layer_metrics
Ptrainable_variables
 ”layer_regularization_losses
Qregularization_losses
R	variables
‘layers
’non_trainable_variables
÷metrics
 
 
 
 
≤
„layer_metrics
Utrainable_variables
 Ўlayer_regularization_losses
Vregularization_losses
W	variables
ўlayers
Џnon_trainable_variables
џmetrics
л
№_query_dense
Ё
_key_dense
ё_value_dense
я_softmax
а_dropout_layer
б_output_dense
$в_self_saveable_object_factories
гtrainable_variables
дregularization_losses
е	variables
ж	keras_api
ќ
зlayer_with_weights-0
зlayer-0
иlayer_with_weights-1
иlayer-1
$й_self_saveable_object_factories
кtrainable_variables
лregularization_losses
м	variables
н	keras_api
Ю
	оaxis

Ьgamma
	Эbeta
$п_self_saveable_object_factories
рtrainable_variables
сregularization_losses
т	variables
у	keras_api
Ю
	фaxis

Юgamma
	Яbeta
$х_self_saveable_object_factories
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
|
$ъ_self_saveable_object_factories
ыtrainable_variables
ьregularization_losses
э	variables
ю	keras_api
|
$€_self_saveable_object_factories
Аtrainable_variables
Бregularization_losses
В	variables
Г	keras_api
 
Ж
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
Ш8
Щ9
Ъ10
Ы11
Ь12
Э13
Ю14
Я15
 
Ж
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
Ш8
Щ9
Ъ10
Ы11
Ь12
Э13
Ю14
Я15
≤
Дlayer_metrics
`trainable_variables
 Еlayer_regularization_losses
aregularization_losses
b	variables
Жlayers
Зnon_trainable_variables
Иmetrics
 
 
 
 
≤
Йlayer_metrics
etrainable_variables
 Кlayer_regularization_losses
fregularization_losses
g	variables
Лlayers
Мnon_trainable_variables
Нmetrics
 
 
 
 
 
≤
Оlayer_metrics
ktrainable_variables
 Пlayer_regularization_losses
lregularization_losses
m	variables
Рlayers
Сnon_trainable_variables
Тmetrics
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

o0
p1
 

o0
p1
≤
Уlayer_metrics
rtrainable_variables
 Фlayer_regularization_losses
sregularization_losses
t	variables
Хlayers
Цnon_trainable_variables
Чmetrics
 
 
 
 
≤
Шlayer_metrics
wtrainable_variables
 Щlayer_regularization_losses
xregularization_losses
y	variables
Ъlayers
Ыnon_trainable_variables
Ьmetrics
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1
 

{0
|1
≥
Эlayer_metrics
~trainable_variables
 Юlayer_regularization_losses
regularization_losses
А	variables
Яlayers
†non_trainable_variables
°metrics
 
 
 
 
µ
Ґlayer_metrics
Гtrainable_variables
 £layer_regularization_losses
Дregularization_losses
Е	variables
§layers
•non_trainable_variables
¶metrics
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

З0
И1
 

З0
И1
µ
Іlayer_metrics
Кtrainable_variables
 ®layer_regularization_losses
Лregularization_losses
М	variables
©layers
™non_trainable_variables
Ђmetrics
{y
VARIABLE_VALUE5token_and_position_embedding_1/embedding_2/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5token_and_position_embedding_1/embedding_3/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE7transformer_block_3/multi_head_attention_3/query/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE5transformer_block_3/multi_head_attention_3/query/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE5transformer_block_3/multi_head_attention_3/key/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE3transformer_block_3/multi_head_attention_3/key/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE7transformer_block_3/multi_head_attention_3/value/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE5transformer_block_3/multi_head_attention_3/value/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUEBtransformer_block_3/multi_head_attention_3/attention_output/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE@transformer_block_3/multi_head_attention_3/attention_output/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_9/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_9/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_10/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_10/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_3/layer_normalization_6/gamma1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.transformer_block_3/layer_normalization_6/beta1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_3/layer_normalization_7/gamma1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.transformer_block_3/layer_normalization_7/beta1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
 
 
О
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

C0
D1
M2
N3

ђ0
 

О0
 

О0
µ
≠layer_metrics
¶trainable_variables
 Ѓlayer_regularization_losses
Іregularization_losses
®	variables
ѓlayers
∞non_trainable_variables
±metrics
 

П0
 

П0
µ
≤layer_metrics
Ђtrainable_variables
 ≥layer_regularization_losses
ђregularization_losses
≠	variables
іlayers
µnon_trainable_variables
ґmetrics
 
 
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
C0
D1
 
 
 
 

M0
N1
 
 
 
 
 
 
«
Јpartial_output_shape
Єfull_output_shape
Рkernel
	Сbias
$є_self_saveable_object_factories
Їtrainable_variables
їregularization_losses
Љ	variables
љ	keras_api
«
Њpartial_output_shape
њfull_output_shape
Тkernel
	Уbias
$ј_self_saveable_object_factories
Ѕtrainable_variables
¬regularization_losses
√	variables
ƒ	keras_api
«
≈partial_output_shape
∆full_output_shape
Фkernel
	Хbias
$«_self_saveable_object_factories
»trainable_variables
…regularization_losses
 	variables
Ћ	keras_api
|
$ћ_self_saveable_object_factories
Ќtrainable_variables
ќregularization_losses
ѕ	variables
–	keras_api
|
$—_self_saveable_object_factories
“trainable_variables
”regularization_losses
‘	variables
’	keras_api
«
÷partial_output_shape
„full_output_shape
Цkernel
	Чbias
$Ў_self_saveable_object_factories
ўtrainable_variables
Џregularization_losses
џ	variables
№	keras_api
 
@
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
 
@
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
µ
Ёlayer_metrics
гtrainable_variables
 ёlayer_regularization_losses
дregularization_losses
е	variables
яlayers
аnon_trainable_variables
бmetrics
Ф
Шkernel
	Щbias
$в_self_saveable_object_factories
гtrainable_variables
дregularization_losses
е	variables
ж	keras_api
Ф
Ъkernel
	Ыbias
$з_self_saveable_object_factories
иtrainable_variables
йregularization_losses
к	variables
л	keras_api
 
 
Ш0
Щ1
Ъ2
Ы3
 
 
Ш0
Щ1
Ъ2
Ы3
µ
мlayer_metrics
кtrainable_variables
 нlayer_regularization_losses
лregularization_losses
м	variables
оlayers
пnon_trainable_variables
рmetrics
 
 

Ь0
Э1
 

Ь0
Э1
µ
сlayer_metrics
рtrainable_variables
 тlayer_regularization_losses
сregularization_losses
т	variables
уlayers
фnon_trainable_variables
хmetrics
 
 

Ю0
Я1
 

Ю0
Я1
µ
цlayer_metrics
цtrainable_variables
 чlayer_regularization_losses
чregularization_losses
ш	variables
шlayers
щnon_trainable_variables
ъmetrics
 
 
 
 
µ
ыlayer_metrics
ыtrainable_variables
 ьlayer_regularization_losses
ьregularization_losses
э	variables
эlayers
юnon_trainable_variables
€metrics
 
 
 
 
µ
Аlayer_metrics
Аtrainable_variables
 Бlayer_regularization_losses
Бregularization_losses
В	variables
Вlayers
Гnon_trainable_variables
Дmetrics
 
 
*
Y0
Z1
[2
\3
]4
^5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

Еtotal

Жcount
З	variables
И	keras_api
 
 
 
 
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
Р0
С1
 

Р0
С1
µ
Йlayer_metrics
Їtrainable_variables
 Кlayer_regularization_losses
їregularization_losses
Љ	variables
Лlayers
Мnon_trainable_variables
Нmetrics
 
 
 

Т0
У1
 

Т0
У1
µ
Оlayer_metrics
Ѕtrainable_variables
 Пlayer_regularization_losses
¬regularization_losses
√	variables
Рlayers
Сnon_trainable_variables
Тmetrics
 
 
 

Ф0
Х1
 

Ф0
Х1
µ
Уlayer_metrics
»trainable_variables
 Фlayer_regularization_losses
…regularization_losses
 	variables
Хlayers
Цnon_trainable_variables
Чmetrics
 
 
 
 
µ
Шlayer_metrics
Ќtrainable_variables
 Щlayer_regularization_losses
ќregularization_losses
ѕ	variables
Ъlayers
Ыnon_trainable_variables
Ьmetrics
 
 
 
 
µ
Эlayer_metrics
“trainable_variables
 Юlayer_regularization_losses
”regularization_losses
‘	variables
Яlayers
†non_trainable_variables
°metrics
 
 
 

Ц0
Ч1
 

Ц0
Ч1
µ
Ґlayer_metrics
ўtrainable_variables
 £layer_regularization_losses
Џregularization_losses
џ	variables
§layers
•non_trainable_variables
¶metrics
 
 
0
№0
Ё1
ё2
я3
а4
б5
 
 
 

Ш0
Щ1
 

Ш0
Щ1
µ
Іlayer_metrics
гtrainable_variables
 ®layer_regularization_losses
дregularization_losses
е	variables
©layers
™non_trainable_variables
Ђmetrics
 

Ъ0
Ы1
 

Ъ0
Ы1
µ
ђlayer_metrics
иtrainable_variables
 ≠layer_regularization_losses
йregularization_losses
к	variables
Ѓlayers
ѓnon_trainable_variables
∞metrics
 
 

з0
и1
 
 
 
 
 
 
 
 
 
 
 
 
 
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
Е0
Ж1

З	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
|
serving_default_input_3Placeholder*(
_output_shapes
:€€€€€€€€€ДR*
dtype0*
shape:€€€€€€€€€ДR
z
serving_default_input_4Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
э
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_45token_and_position_embedding_1/embedding_3/embeddings5token_and_position_embedding_1/embedding_2/embeddingsconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/beta7transformer_block_3/multi_head_attention_3/query/kernel5transformer_block_3/multi_head_attention_3/query/bias5transformer_block_3/multi_head_attention_3/key/kernel3transformer_block_3/multi_head_attention_3/key/bias7transformer_block_3/multi_head_attention_3/value/kernel5transformer_block_3/multi_head_attention_3/value/biasBtransformer_block_3/multi_head_attention_3/attention_output/kernel@transformer_block_3/multi_head_attention_3/attention_output/bias/transformer_block_3/layer_normalization_6/gamma.transformer_block_3/layer_normalization_6/betadense_9/kerneldense_9/biasdense_10/kerneldense_10/bias/transformer_block_3/layer_normalization_7/gamma.transformer_block_3/layer_normalization_7/betadense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/bias*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_310440
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
∞
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpItoken_and_position_embedding_1/embedding_2/embeddings/Read/ReadVariableOpItoken_and_position_embedding_1/embedding_3/embeddings/Read/ReadVariableOpKtransformer_block_3/multi_head_attention_3/query/kernel/Read/ReadVariableOpItransformer_block_3/multi_head_attention_3/query/bias/Read/ReadVariableOpItransformer_block_3/multi_head_attention_3/key/kernel/Read/ReadVariableOpGtransformer_block_3/multi_head_attention_3/key/bias/Read/ReadVariableOpKtransformer_block_3/multi_head_attention_3/value/kernel/Read/ReadVariableOpItransformer_block_3/multi_head_attention_3/value/bias/Read/ReadVariableOpVtransformer_block_3/multi_head_attention_3/attention_output/kernel/Read/ReadVariableOpTtransformer_block_3/multi_head_attention_3/attention_output/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOpCtransformer_block_3/layer_normalization_6/gamma/Read/ReadVariableOpBtransformer_block_3/layer_normalization_6/beta/Read/ReadVariableOpCtransformer_block_3/layer_normalization_7/gamma/Read/ReadVariableOpBtransformer_block_3/layer_normalization_7/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*3
Tin,
*2(*
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_311969
≥
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/bias5token_and_position_embedding_1/embedding_2/embeddings5token_and_position_embedding_1/embedding_3/embeddings7transformer_block_3/multi_head_attention_3/query/kernel5transformer_block_3/multi_head_attention_3/query/bias5transformer_block_3/multi_head_attention_3/key/kernel3transformer_block_3/multi_head_attention_3/key/bias7transformer_block_3/multi_head_attention_3/value/kernel5transformer_block_3/multi_head_attention_3/value/biasBtransformer_block_3/multi_head_attention_3/attention_output/kernel@transformer_block_3/multi_head_attention_3/attention_output/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias/transformer_block_3/layer_normalization_6/gamma.transformer_block_3/layer_normalization_6/beta/transformer_block_3/layer_normalization_7/gamma.transformer_block_3/layer_normalization_7/betatotalcount*2
Tin+
)2'*
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_312093ып
Ъ
ч
D__inference_conv1d_2_layer_call_and_return_conditional_losses_311101

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR *
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR *
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€ДR 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€ДR ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ДR 
 
_user_specified_nameinputs
®
Z
.__inference_concatenate_1_layer_call_fn_311499
inputs_0
inputs_1
identityЎ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_3097712
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€а:€€€€€€€€€:R N
(
_output_shapes
:€€€€€€€€€а
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
Џ
§
(__inference_model_1_layer_call_fn_311085
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
identityИҐStatefulPartitionedCall÷
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
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_3102852
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::22
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
вU
ъ
__inference__traced_save_311969
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
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableopT
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
 savev2_count_read_readvariableop
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
ShardedFilenameЭ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*ѓ
value•BҐ'B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names÷
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices÷
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableopPsavev2_token_and_position_embedding_1_embedding_2_embeddings_read_readvariableopPsavev2_token_and_position_embedding_1_embedding_3_embeddings_read_readvariableopRsavev2_transformer_block_3_multi_head_attention_3_query_kernel_read_readvariableopPsavev2_transformer_block_3_multi_head_attention_3_query_bias_read_readvariableopPsavev2_transformer_block_3_multi_head_attention_3_key_kernel_read_readvariableopNsavev2_transformer_block_3_multi_head_attention_3_key_bias_read_readvariableopRsavev2_transformer_block_3_multi_head_attention_3_value_kernel_read_readvariableopPsavev2_transformer_block_3_multi_head_attention_3_value_bias_read_readvariableop]savev2_transformer_block_3_multi_head_attention_3_attention_output_kernel_read_readvariableop[savev2_transformer_block_3_multi_head_attention_3_attention_output_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableopJsavev2_transformer_block_3_layer_normalization_6_gamma_read_readvariableopIsavev2_transformer_block_3_layer_normalization_6_beta_read_readvariableopJsavev2_transformer_block_3_layer_normalization_7_gamma_read_readvariableopIsavev2_transformer_block_3_layer_normalization_7_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *5
dtypes+
)2'2
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

identity_1Identity_1:output:0*ѕ
_input_shapesљ
Ї: :  : :	  : : : : : : : : : :	и@:@:@@:@:@:: :	ДR :  : :  : :  : :  : : @:@:@ : : : : : : : : 2(
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
:	и@: 
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
::$ 

_output_shapes

: :%!

_output_shapes
:	ДR :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:@ :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
я
~
)__inference_dense_13_layer_call_fn_311612

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallч
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
GPU2*0J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_3099042
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
Ё
ю
H__inference_sequential_3_layer_call_and_return_conditional_losses_310596

inputs
dense_9_310585
dense_9_310587
dense_10_310590
dense_10_310592
identityИҐ dense_10/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallЦ
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_310585dense_9_310587*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3104752!
dense_9/StatefulPartitionedCallљ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_310590dense_10_310592*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_3105212"
 dense_10/StatefulPartitionedCall∆
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
м
©
6__inference_batch_normalization_2_layer_call_fn_311286

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
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3092592
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Щ
G
+__inference_dropout_11_layer_call_fn_311593

inputs
identity«
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3098812
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
•
d
+__inference_dropout_10_layer_call_fn_311541

inputs
identityИҐStatefulPartitionedCallя
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3098192
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
∞ 
в
C__inference_dense_9_layer_call_and_return_conditional_losses_310475

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
ReluЮ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€# ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
у0
»
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311253

inputs
assignmovingavg_311228
assignmovingavg_1_311234)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/311228*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_311228*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/311228*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/311228*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_311228AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/311228*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/311234*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_311234*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/311234*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/311234*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_311234AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/311234*
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
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ё
ю
H__inference_sequential_3_layer_call_and_return_conditional_losses_310569

inputs
dense_9_310558
dense_9_310560
dense_10_310563
dense_10_310565
identityИҐ dense_10/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallЦ
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_310558dense_9_310560*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3104752!
dense_9/StatefulPartitionedCallљ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_310563dense_10_310565*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_3105212"
 dense_10/StatefulPartitionedCall∆
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
”
y
)__inference_restored_function_body_305628
x
unknown
	unknown_0
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *c
f^R\
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_3010332
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€ДR 2

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
о	
Ё
D__inference_dense_12_layer_call_and_return_conditional_losses_309848

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
ў№
÷
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_301540

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
identityИҐ.layer_normalization_6/batchnorm/ReadVariableOpҐ2layer_normalization_6/batchnorm/mul/ReadVariableOpҐ.layer_normalization_7/batchnorm/ReadVariableOpҐ2layer_normalization_7/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_3/attention_output/add/ReadVariableOpҐDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_3/key/add/ReadVariableOpҐ7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_3/query/add/ReadVariableOpҐ9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_3/value/add/ReadVariableOpҐ9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpҐ,sequential_3/dense_10/BiasAdd/ReadVariableOpҐ.sequential_3/dense_10/Tensordot/ReadVariableOpҐ+sequential_3/dense_9/BiasAdd/ReadVariableOpҐ-sequential_3/dense_9/Tensordot/ReadVariableOpэ
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/Einsumџ
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOpх
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2"
 multi_head_attention_3/query/addч
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/Einsum’
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOpн
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2 
multi_head_attention_3/key/addэ
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/Einsumџ
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOpх
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2"
 multi_head_attention_3/value/addБ
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_3/Mul/y∆
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€# 2
multi_head_attention_3/Mulь
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/Einsumƒ
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€##2(
&multi_head_attention_3/softmax/Softmax 
'multi_head_attention_3/dropout/IdentityIdentity0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€##2)
'multi_head_attention_3/dropout/IdentityФ
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/Identity:output:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/EinsumЮ
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsumш
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOpЭ
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2-
+multi_head_attention_3/attention_output/addЫ
dropout_8/IdentityIdentity/multi_head_attention_3/attention_output/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_8/Identityn
addAddV2inputsdropout_8/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
addґ
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indicesя
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2$
"layer_normalization_6/moments/meanЋ
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2,
*layer_normalization_6/moments/StopGradientл
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 21
/layer_normalization_6/moments/SquaredDifferenceЊ
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indicesЧ
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2(
&layer_normalization_6/moments/varianceУ
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_6/batchnorm/add/yк
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2%
#layer_normalization_6/batchnorm/addґ
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€#2'
%layer_normalization_6/batchnorm/Rsqrtа
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOpо
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_6/batchnorm/mulљ
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/mul_1б
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/mul_2‘
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOpк
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_6/batchnorm/subб
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/add_1’
-sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_3/dense_9/Tensordot/ReadVariableOpФ
#sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_3/dense_9/Tensordot/axesЫ
#sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_3/dense_9/Tensordot/free•
$sequential_3/dense_9/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/ShapeЮ
,sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/GatherV2/axisЇ
'sequential_3/dense_9/Tensordot/GatherV2GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/free:output:05sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/GatherV2Ґ
.sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_3/dense_9/Tensordot/GatherV2_1/axisј
)sequential_3/dense_9/Tensordot/GatherV2_1GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/axes:output:07sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_3/dense_9/Tensordot/GatherV2_1Ц
$sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_3/dense_9/Tensordot/Const‘
#sequential_3/dense_9/Tensordot/ProdProd0sequential_3/dense_9/Tensordot/GatherV2:output:0-sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_3/dense_9/Tensordot/ProdЪ
&sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_3/dense_9/Tensordot/Const_1№
%sequential_3/dense_9/Tensordot/Prod_1Prod2sequential_3/dense_9/Tensordot/GatherV2_1:output:0/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_3/dense_9/Tensordot/Prod_1Ъ
*sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_3/dense_9/Tensordot/concat/axisЩ
%sequential_3/dense_9/Tensordot/concatConcatV2,sequential_3/dense_9/Tensordot/free:output:0,sequential_3/dense_9/Tensordot/axes:output:03sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_9/Tensordot/concatа
$sequential_3/dense_9/Tensordot/stackPack,sequential_3/dense_9/Tensordot/Prod:output:0.sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/stackт
(sequential_3/dense_9/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0.sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2*
(sequential_3/dense_9/Tensordot/transposeу
&sequential_3/dense_9/Tensordot/ReshapeReshape,sequential_3/dense_9/Tensordot/transpose:y:0-sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_3/dense_9/Tensordot/Reshapeт
%sequential_3/dense_9/Tensordot/MatMulMatMul/sequential_3/dense_9/Tensordot/Reshape:output:05sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_3/dense_9/Tensordot/MatMulЪ
&sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_3/dense_9/Tensordot/Const_2Ю
,sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/concat_1/axis¶
'sequential_3/dense_9/Tensordot/concat_1ConcatV20sequential_3/dense_9/Tensordot/GatherV2:output:0/sequential_3/dense_9/Tensordot/Const_2:output:05sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/concat_1д
sequential_3/dense_9/TensordotReshape/sequential_3/dense_9/Tensordot/MatMul:product:00sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2 
sequential_3/dense_9/TensordotЋ
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOpџ
sequential_3/dense_9/BiasAddBiasAdd'sequential_3/dense_9/Tensordot:output:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
sequential_3/dense_9/BiasAddЫ
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
sequential_3/dense_9/ReluЎ
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOpЦ
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axesЭ
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free•
%sequential_3/dense_10/Tensordot/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape†
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axisњ
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2§
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axis≈
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1Ш
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/ConstЎ
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/ProdЬ
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1а
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1Ь
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axisЮ
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concatд
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stackу
)sequential_3/dense_10/Tensordot/transpose	Transpose'sequential_3/dense_9/Relu:activations:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2+
)sequential_3/dense_10/Tensordot/transposeч
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_3/dense_10/Tensordot/Reshapeц
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_3/dense_10/Tensordot/MatMulЬ
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_2†
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axisЂ
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1и
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2!
sequential_3/dense_10/Tensordotќ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOpя
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
sequential_3/dense_10/BiasAddТ
dropout_9/IdentityIdentity&sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_9/IdentityХ
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
add_1ґ
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indicesб
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2$
"layer_normalization_7/moments/meanЋ
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2,
*layer_normalization_7/moments/StopGradientн
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 21
/layer_normalization_7/moments/SquaredDifferenceЊ
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indicesЧ
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2(
&layer_normalization_7/moments/varianceУ
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_7/batchnorm/add/yк
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2%
#layer_normalization_7/batchnorm/addґ
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€#2'
%layer_normalization_7/batchnorm/Rsqrtа
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOpо
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_7/batchnorm/mulњ
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/mul_1б
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/mul_2‘
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOpк
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_7/batchnorm/subб
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/add_1’
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp.^sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2М
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
:€€€€€€€€€# 
 
_user_specified_nameinputs
ђ
R
&__inference_add_1_layer_call_fn_311475
inputs_0
inputs_1
identity”
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3097082
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:€€€€€€€€€# :€€€€€€€€€# :U Q
+
_output_shapes
:€€€€€€€€€# 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€# 
"
_user_specified_name
inputs/1
Ќ
І
-__inference_sequential_3_layer_call_fn_310607
dense_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCalldense_9_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3105962
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€# 
'
_user_specified_namedense_9_input
с	
Ё
D__inference_dense_11_layer_call_and_return_conditional_losses_309791

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	и@*
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
:€€€€€€€€€и::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
±I
Ђ
H__inference_sequential_3_layer_call_and_return_conditional_losses_311726

inputs-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*dense_10_tensordot_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource
identityИҐdense_10/BiasAdd/ReadVariableOpҐ!dense_10/Tensordot/ReadVariableOpҐdense_9/BiasAdd/ReadVariableOpҐ dense_9/Tensordot/ReadVariableOpЃ
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
dense_9/Tensordot/axesБ
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
dense_9/Tensordot/ShapeД
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisщ
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2И
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axis€
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
dense_9/Tensordot/Const†
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/ProdА
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1®
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1А
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisЎ
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatђ
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stack®
dense_9/Tensordot/transpose	Transposeinputs!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dense_9/Tensordot/transposeњ
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_9/Tensordot/ReshapeЊ
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_9/Tensordot/MatMulА
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_9/Tensordot/Const_2Д
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisе
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1∞
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
dense_9/Tensordot§
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_9/BiasAdd/ReadVariableOpІ
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
dense_9/BiasAddt
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
dense_9/Relu±
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
dense_10/Tensordot/axesГ
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
dense_10/Tensordot/ShapeЖ
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axisю
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2К
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axisД
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
dense_10/Tensordot/Const§
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/ProdВ
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1ђ
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1В
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axisЁ
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat∞
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stackњ
dense_10/Tensordot/transpose	Transposedense_9/Relu:activations:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
dense_10/Tensordot/transpose√
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_10/Tensordot/Reshape¬
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_10/Tensordot/MatMulВ
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_2Ж
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisк
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1і
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dense_10/TensordotІ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_10/BiasAdd/ReadVariableOpЂ
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dense_10/BiasAddы
IdentityIdentitydense_10/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
т
Е
H__inference_sequential_3_layer_call_and_return_conditional_losses_310552
dense_9_input
dense_9_310541
dense_9_310543
dense_10_310546
dense_10_310548
identityИҐ dense_10/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallЭ
dense_9/StatefulPartitionedCallStatefulPartitionedCalldense_9_inputdense_9_310541dense_9_310543*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3104752!
dense_9/StatefulPartitionedCallљ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_310546dense_10_310548*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_3105212"
 dense_10/StatefulPartitionedCall∆
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€# 
'
_user_specified_namedense_9_input
н
}
(__inference_dense_9_layer_call_fn_311792

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3104752
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€# ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
ИX
Ю
C__inference_model_1_layer_call_and_return_conditional_losses_310285

inputs
inputs_1)
%token_and_position_embedding_1_310195)
%token_and_position_embedding_1_310197
conv1d_2_310200
conv1d_2_310202
conv1d_3_310206
conv1d_3_310208 
batch_normalization_2_310213 
batch_normalization_2_310215 
batch_normalization_2_310217 
batch_normalization_2_310219 
batch_normalization_3_310222 
batch_normalization_3_310224 
batch_normalization_3_310226 
batch_normalization_3_310228
transformer_block_3_310232
transformer_block_3_310234
transformer_block_3_310236
transformer_block_3_310238
transformer_block_3_310240
transformer_block_3_310242
transformer_block_3_310244
transformer_block_3_310246
transformer_block_3_310248
transformer_block_3_310250
transformer_block_3_310252
transformer_block_3_310254
transformer_block_3_310256
transformer_block_3_310258
transformer_block_3_310260
transformer_block_3_310262
dense_11_310267
dense_11_310269
dense_12_310273
dense_12_310275
dense_13_310279
dense_13_310281
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ conv1d_2/StatefulPartitionedCallҐ conv1d_3/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ6token_and_position_embedding_1/StatefulPartitionedCallҐ+transformer_block_3/StatefulPartitionedCallў
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_1_310195%token_and_position_embedding_1_310197*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_30562828
6token_and_position_embedding_1/StatefulPartitionedCall’
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0conv1d_2_310200conv1d_2_310202*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3094692"
 conv1d_2/StatefulPartitionedCall†
#average_pooling1d_3/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ё * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_3091272%
#average_pooling1d_3/PartitionedCall¬
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_3/PartitionedCall:output:0conv1d_3_310206conv1d_3_310208*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ё *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_3095022"
 conv1d_3/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_3091572%
#average_pooling1d_5/PartitionedCallЯ
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_3091422%
#average_pooling1d_4/PartitionedCall¬
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_2_310213batch_normalization_2_310215batch_normalization_2_310217batch_normalization_2_310219*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3095752/
-batch_normalization_2/StatefulPartitionedCall¬
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_3_310222batch_normalization_3_310224batch_normalization_3_310226batch_normalization_3_310228*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3096662/
-batch_normalization_3/StatefulPartitionedCallї
add_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:06batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3097082
add_1/PartitionedCallи
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0transformer_block_3_310232transformer_block_3_310234transformer_block_3_310236transformer_block_3_310238transformer_block_3_310240transformer_block_3_310242transformer_block_3_310244transformer_block_3_310246transformer_block_3_310248transformer_block_3_310250transformer_block_3_310252transformer_block_3_310254transformer_block_3_310256transformer_block_3_310258transformer_block_3_310260transformer_block_3_310262*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_3057392-
+transformer_block_3/StatefulPartitionedCallЙ
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€а* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_3097562
flatten_1/PartitionedCallО
concatenate_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_3097712
concatenate_1/PartitionedCallЈ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_11_310267dense_11_310269*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_3097912"
 dense_11/StatefulPartitionedCallА
dropout_10/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3098242
dropout_10/PartitionedCallі
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_12_310273dense_12_310275*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3098482"
 dense_12/StatefulPartitionedCallА
dropout_11/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3098812
dropout_11/PartitionedCallі
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_13_310279dense_13_310281*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_3099042"
 dense_13/StatefulPartitionedCallу
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ДR
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ЊК
б
C__inference_model_1_layer_call_and_return_conditional_losses_310791
inputs_0
inputs_1)
%token_and_position_embedding_1_310611)
%token_and_position_embedding_1_3106138
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource0
,batch_normalization_2_assignmovingavg_3106602
.batch_normalization_2_assignmovingavg_1_310666?
;batch_normalization_2_batchnorm_mul_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource0
,batch_normalization_3_assignmovingavg_3106922
.batch_normalization_3_assignmovingavg_1_310698?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource
transformer_block_3_310717
transformer_block_3_310719
transformer_block_3_310721
transformer_block_3_310723
transformer_block_3_310725
transformer_block_3_310727
transformer_block_3_310729
transformer_block_3_310731
transformer_block_3_310733
transformer_block_3_310735
transformer_block_3_310737
transformer_block_3_310739
transformer_block_3_310741
transformer_block_3_310743
transformer_block_3_310745
transformer_block_3_310747+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identityИҐ9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_2/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_3/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐconv1d_2/BiasAdd/ReadVariableOpҐ+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpҐconv1d_3/BiasAdd/ReadVariableOpҐ+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpҐdense_11/BiasAdd/ReadVariableOpҐdense_11/MatMul/ReadVariableOpҐdense_12/BiasAdd/ReadVariableOpҐdense_12/MatMul/ReadVariableOpҐdense_13/BiasAdd/ReadVariableOpҐdense_13/MatMul/ReadVariableOpҐ6token_and_position_embedding_1/StatefulPartitionedCallҐ+transformer_block_3/StatefulPartitionedCallџ
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_0%token_and_position_embedding_1_310611%token_and_position_embedding_1_310613*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_30562828
6token_and_position_embedding_1/StatefulPartitionedCallЛ
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2 
conv1d_2/conv1d/ExpandDims/dimл
conv1d_2/conv1d/ExpandDims
ExpandDims?token_and_position_embedding_1/StatefulPartitionedCall:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2
conv1d_2/conv1d/ExpandDims”
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimџ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_2/conv1d/ExpandDims_1џ
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR *
paddingSAME*
strides
2
conv1d_2/conv1dЃ
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR *
squeeze_dims

э€€€€€€€€2
conv1d_2/conv1d/SqueezeІ
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_2/BiasAdd/ReadVariableOp±
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2
conv1d_2/BiasAddx
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2
conv1d_2/ReluК
"average_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_3/ExpandDims/dim”
average_pooling1d_3/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0+average_pooling1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2 
average_pooling1d_3/ExpandDimsе
average_pooling1d_3/AvgPoolAvgPool'average_pooling1d_3/ExpandDims:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё *
ksize
*
paddingVALID*
strides
2
average_pooling1d_3/AvgPoolє
average_pooling1d_3/SqueezeSqueeze$average_pooling1d_3/AvgPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё *
squeeze_dims
2
average_pooling1d_3/SqueezeЛ
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2 
conv1d_3/conv1d/ExpandDims/dim–
conv1d_3/conv1d/ExpandDims
ExpandDims$average_pooling1d_3/Squeeze:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё 2
conv1d_3/conv1d/ExpandDims”
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimџ
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_3/conv1d/ExpandDims_1џ
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё *
paddingSAME*
strides
2
conv1d_3/conv1dЃ
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё *
squeeze_dims

э€€€€€€€€2
conv1d_3/conv1d/SqueezeІ
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_3/BiasAdd/ReadVariableOp±
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ё 2
conv1d_3/BiasAddx
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё 2
conv1d_3/ReluК
"average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_5/ExpandDims/dimч
average_pooling1d_5/ExpandDims
ExpandDims?token_and_position_embedding_1/StatefulPartitionedCall:output:0+average_pooling1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2 
average_pooling1d_5/ExpandDimsж
average_pooling1d_5/AvgPoolAvgPool'average_pooling1d_5/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€# *
ksize	
ђ*
paddingVALID*
strides	
ђ2
average_pooling1d_5/AvgPoolЄ
average_pooling1d_5/SqueezeSqueeze$average_pooling1d_5/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€# *
squeeze_dims
2
average_pooling1d_5/SqueezeК
"average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_4/ExpandDims/dim”
average_pooling1d_4/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0+average_pooling1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё 2 
average_pooling1d_4/ExpandDimsд
average_pooling1d_4/AvgPoolAvgPool'average_pooling1d_4/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€# *
ksize

*
paddingVALID*
strides

2
average_pooling1d_4/AvgPoolЄ
average_pooling1d_4/SqueezeSqueeze$average_pooling1d_4/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€# *
squeeze_dims
2
average_pooling1d_4/Squeezeљ
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indicesу
"batch_normalization_2/moments/meanMean$average_pooling1d_4/Squeeze:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_2/moments/mean¬
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_2/moments/StopGradientИ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference$average_pooling1d_4/Squeeze:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 21
/batch_normalization_2/moments/SquaredDifference≈
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indicesО
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_2/moments/variance√
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_2/moments/SqueezeЋ
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1О
+batch_normalization_2/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/310660*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_2/AssignMovingAvg/decay’
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_2_assignmovingavg_310660*
_output_shapes
: *
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpя
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/310660*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/sub÷
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/310660*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/mul≥
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_2_assignmovingavg_310660-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg/310660*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpФ
-batch_normalization_2/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/310666*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_2/AssignMovingAvg_1/decayџ
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_2_assignmovingavg_1_310666*
_output_shapes
: *
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpй
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/310666*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/subа
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/310666*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/mulњ
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_2_assignmovingavg_1_310666/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_2/AssignMovingAvg_1/310666*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yЏ
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/add•
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_2/batchnorm/Rsqrtа
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/mulЏ
%batch_normalization_2/batchnorm/mul_1Mul$average_pooling1d_4/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%batch_normalization_2/batchnorm/mul_1”
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_2/batchnorm/mul_2‘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpў
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/subб
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%batch_normalization_2/batchnorm/add_1љ
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_3/moments/mean/reduction_indicesу
"batch_normalization_3/moments/meanMean$average_pooling1d_5/Squeeze:output:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_3/moments/mean¬
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_3/moments/StopGradientИ
/batch_normalization_3/moments/SquaredDifferenceSquaredDifference$average_pooling1d_5/Squeeze:output:03batch_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 21
/batch_normalization_3/moments/SquaredDifference≈
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_3/moments/variance/reduction_indicesО
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_3/moments/variance√
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_3/moments/SqueezeЋ
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1О
+batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/310692*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_3/AssignMovingAvg/decay’
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_3_assignmovingavg_310692*
_output_shapes
: *
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpя
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/310692*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/sub÷
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/310692*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/mul≥
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_3_assignmovingavg_310692-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_3/AssignMovingAvg/310692*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpФ
-batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/310698*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_3/AssignMovingAvg_1/decayџ
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_3_assignmovingavg_1_310698*
_output_shapes
: *
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpй
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/310698*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/subа
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/310698*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/mulњ
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_3_assignmovingavg_1_310698/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_3/AssignMovingAvg_1/310698*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yЏ
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/mulЏ
%batch_normalization_3/batchnorm/mul_1Mul$average_pooling1d_5/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%batch_normalization_3/batchnorm/mul_1”
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/mul_2‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpў
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/subб
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%batch_normalization_3/batchnorm/add_1Ђ
	add_1/addAddV2)batch_normalization_2/batchnorm/add_1:z:0)batch_normalization_3/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
	add_1/add„
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/add:z:0transformer_block_3_310717transformer_block_3_310719transformer_block_3_310721transformer_block_3_310723transformer_block_3_310725transformer_block_3_310727transformer_block_3_310729transformer_block_3_310731transformer_block_3_310733transformer_block_3_310735transformer_block_3_310737transformer_block_3_310739transformer_block_3_310741transformer_block_3_310743transformer_block_3_310745transformer_block_3_310747*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_3064352-
+transformer_block_3/StatefulPartitionedCalls
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€`  2
flatten_1/Constі
flatten_1/ReshapeReshape4transformer_block_3/StatefulPartitionedCall:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€а2
flatten_1/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisЊ
concatenate_1/concatConcatV2flatten_1/Reshape:output:0inputs_1"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€и2
concatenate_1/concat©
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	и@*
dtype02 
dense_11/MatMul/ReadVariableOp•
dense_11/MatMulMatMulconcatenate_1/concat:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_11/MatMulІ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_11/BiasAdd/ReadVariableOp•
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_11/BiasAdds
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_11/Reluy
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_10/dropout/Const©
dropout_10/dropout/MulMuldense_11/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_10/dropout/Mul
dropout_10/dropout/ShapeShapedense_11/Relu:activations:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape’
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype021
/dropout_10/dropout/random_uniform/RandomUniformЛ
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_10/dropout/GreaterEqual/yк
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
dropout_10/dropout/GreaterEqual†
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_10/dropout/Cast¶
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_10/dropout/Mul_1®
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_12/MatMul/ReadVariableOp§
dense_12/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/MatMulІ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp•
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/Reluy
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_11/dropout/Const©
dropout_11/dropout/MulMuldense_12/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape’
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype021
/dropout_11/dropout/random_uniform/RandomUniformЛ
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_11/dropout/GreaterEqual/yк
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
dropout_11/dropout/GreaterEqual†
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_11/dropout/Cast¶
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_11/dropout/Mul_1®
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_13/MatMul/ReadVariableOp§
dense_13/MatMulMatMuldropout_11/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_13/MatMulІ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp•
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_13/BiasAddЁ	
IdentityIdentitydense_13/BiasAdd:output:0:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::2v
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
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:R N
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
»
©
6__inference_batch_normalization_3_layer_call_fn_311450

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3096462
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
у0
»
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_309259

inputs
assignmovingavg_309234
assignmovingavg_1_309240)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/309234*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_309234*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/309234*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/309234*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_309234AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/309234*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/309240*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_309240*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/309240*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/309240*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_309240AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/309240*
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
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Д
P
4__inference_average_pooling1d_5_layer_call_fn_309163

inputs
identityж
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
GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_3091572
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
Ъ
ч
D__inference_conv1d_2_layer_call_and_return_conditional_losses_309469

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR *
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR *
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€ДR 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€ДR ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ДR 
 
_user_specified_nameinputs
и
И
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_309666

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/add_1я
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
т
Е
H__inference_sequential_3_layer_call_and_return_conditional_losses_310538
dense_9_input
dense_9_310486
dense_9_310488
dense_10_310532
dense_10_310534
identityИҐ dense_10/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallЭ
dense_9/StatefulPartitionedCallStatefulPartitionedCalldense_9_inputdense_9_310486dense_9_310488*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3104752!
dense_9/StatefulPartitionedCallљ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_310532dense_10_310534*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_3105212"
 dense_10/StatefulPartitionedCall∆
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€# 
'
_user_specified_namedense_9_input
Є
†
-__inference_sequential_3_layer_call_fn_311752

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
 *+
_output_shapes
:€€€€€€€€€# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3105962
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
µ
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_309756

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€а2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€а2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€# :S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
÷
§
(__inference_model_1_layer_call_fn_311007
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
identityИҐStatefulPartitionedCall“
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
:€€€€€€€€€*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_3101132
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::22
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
и
И
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311191

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/add_1я
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
ч
k
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_309157

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
ђ*
paddingVALID*
strides	
ђ2	
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
»
©
6__inference_batch_normalization_2_layer_call_fn_311204

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3095552
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
…
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_311536

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
Љ0
»
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311171

inputs
assignmovingavg_311146
assignmovingavg_1_311152)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/311146*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_311146*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/311146*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/311146*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_311146AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/311146*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/311152*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_311152*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/311152*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/311152*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_311152AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/311152*
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
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/add_1Ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
В
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_309819

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
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
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
Иэ
÷
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_303054

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
identityИҐ.layer_normalization_6/batchnorm/ReadVariableOpҐ2layer_normalization_6/batchnorm/mul/ReadVariableOpҐ.layer_normalization_7/batchnorm/ReadVariableOpҐ2layer_normalization_7/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_3/attention_output/add/ReadVariableOpҐDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_3/key/add/ReadVariableOpҐ7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_3/query/add/ReadVariableOpҐ9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_3/value/add/ReadVariableOpҐ9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpҐ,sequential_3/dense_10/BiasAdd/ReadVariableOpҐ.sequential_3/dense_10/Tensordot/ReadVariableOpҐ+sequential_3/dense_9/BiasAdd/ReadVariableOpҐ-sequential_3/dense_9/Tensordot/ReadVariableOpэ
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/Einsumџ
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOpх
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2"
 multi_head_attention_3/query/addч
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/Einsum’
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOpн
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2 
multi_head_attention_3/key/addэ
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/Einsumџ
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOpх
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2"
 multi_head_attention_3/value/addБ
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_3/Mul/y∆
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€# 2
multi_head_attention_3/Mulь
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/Einsumƒ
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€##2(
&multi_head_attention_3/softmax/Softmax°
,multi_head_attention_3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,multi_head_attention_3/dropout/dropout/ConstВ
*multi_head_attention_3/dropout/dropout/MulMul0multi_head_attention_3/softmax/Softmax:softmax:05multi_head_attention_3/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€##2,
*multi_head_attention_3/dropout/dropout/MulЉ
,multi_head_attention_3/dropout/dropout/ShapeShape0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_3/dropout/dropout/ShapeЩ
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_3/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€##*
dtype02E
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_3/dropout/dropout/GreaterEqual/y¬
3multi_head_attention_3/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_3/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€##25
3multi_head_attention_3/dropout/dropout/GreaterEqualд
+multi_head_attention_3/dropout/dropout/CastCast7multi_head_attention_3/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€##2-
+multi_head_attention_3/dropout/dropout/Castю
,multi_head_attention_3/dropout/dropout/Mul_1Mul.multi_head_attention_3/dropout/dropout/Mul:z:0/multi_head_attention_3/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€##2.
,multi_head_attention_3/dropout/dropout/Mul_1Ф
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/dropout/Mul_1:z:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/EinsumЮ
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsumш
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOpЭ
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2-
+multi_head_attention_3/attention_output/addw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_8/dropout/ConstЊ
dropout_8/dropout/MulMul/multi_head_attention_3/attention_output/add:z:0 dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_8/dropout/MulС
dropout_8/dropout/ShapeShape/multi_head_attention_3/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape÷
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€# *
dtype020
.dropout_8/dropout/random_uniform/RandomUniformЙ
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_8/dropout/GreaterEqual/yк
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2 
dropout_8/dropout/GreaterEqual°
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€# 2
dropout_8/dropout/Cast¶
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_8/dropout/Mul_1n
addAddV2inputsdropout_8/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
addґ
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indicesя
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2$
"layer_normalization_6/moments/meanЋ
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2,
*layer_normalization_6/moments/StopGradientл
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 21
/layer_normalization_6/moments/SquaredDifferenceЊ
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indicesЧ
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2(
&layer_normalization_6/moments/varianceУ
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_6/batchnorm/add/yк
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2%
#layer_normalization_6/batchnorm/addґ
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€#2'
%layer_normalization_6/batchnorm/Rsqrtа
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOpо
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_6/batchnorm/mulљ
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/mul_1б
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/mul_2‘
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOpк
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_6/batchnorm/subб
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/add_1’
-sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_3/dense_9/Tensordot/ReadVariableOpФ
#sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_3/dense_9/Tensordot/axesЫ
#sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_3/dense_9/Tensordot/free•
$sequential_3/dense_9/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/ShapeЮ
,sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/GatherV2/axisЇ
'sequential_3/dense_9/Tensordot/GatherV2GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/free:output:05sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/GatherV2Ґ
.sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_3/dense_9/Tensordot/GatherV2_1/axisј
)sequential_3/dense_9/Tensordot/GatherV2_1GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/axes:output:07sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_3/dense_9/Tensordot/GatherV2_1Ц
$sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_3/dense_9/Tensordot/Const‘
#sequential_3/dense_9/Tensordot/ProdProd0sequential_3/dense_9/Tensordot/GatherV2:output:0-sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_3/dense_9/Tensordot/ProdЪ
&sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_3/dense_9/Tensordot/Const_1№
%sequential_3/dense_9/Tensordot/Prod_1Prod2sequential_3/dense_9/Tensordot/GatherV2_1:output:0/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_3/dense_9/Tensordot/Prod_1Ъ
*sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_3/dense_9/Tensordot/concat/axisЩ
%sequential_3/dense_9/Tensordot/concatConcatV2,sequential_3/dense_9/Tensordot/free:output:0,sequential_3/dense_9/Tensordot/axes:output:03sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_9/Tensordot/concatа
$sequential_3/dense_9/Tensordot/stackPack,sequential_3/dense_9/Tensordot/Prod:output:0.sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/stackт
(sequential_3/dense_9/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0.sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2*
(sequential_3/dense_9/Tensordot/transposeу
&sequential_3/dense_9/Tensordot/ReshapeReshape,sequential_3/dense_9/Tensordot/transpose:y:0-sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_3/dense_9/Tensordot/Reshapeт
%sequential_3/dense_9/Tensordot/MatMulMatMul/sequential_3/dense_9/Tensordot/Reshape:output:05sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_3/dense_9/Tensordot/MatMulЪ
&sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_3/dense_9/Tensordot/Const_2Ю
,sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/concat_1/axis¶
'sequential_3/dense_9/Tensordot/concat_1ConcatV20sequential_3/dense_9/Tensordot/GatherV2:output:0/sequential_3/dense_9/Tensordot/Const_2:output:05sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/concat_1д
sequential_3/dense_9/TensordotReshape/sequential_3/dense_9/Tensordot/MatMul:product:00sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2 
sequential_3/dense_9/TensordotЋ
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOpџ
sequential_3/dense_9/BiasAddBiasAdd'sequential_3/dense_9/Tensordot:output:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
sequential_3/dense_9/BiasAddЫ
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
sequential_3/dense_9/ReluЎ
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOpЦ
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axesЭ
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free•
%sequential_3/dense_10/Tensordot/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape†
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axisњ
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2§
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axis≈
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1Ш
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/ConstЎ
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/ProdЬ
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1а
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1Ь
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axisЮ
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concatд
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stackу
)sequential_3/dense_10/Tensordot/transpose	Transpose'sequential_3/dense_9/Relu:activations:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2+
)sequential_3/dense_10/Tensordot/transposeч
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_3/dense_10/Tensordot/Reshapeц
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_3/dense_10/Tensordot/MatMulЬ
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_2†
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axisЂ
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1и
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2!
sequential_3/dense_10/Tensordotќ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOpя
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
sequential_3/dense_10/BiasAddw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_9/dropout/Constµ
dropout_9/dropout/MulMul&sequential_3/dense_10/BiasAdd:output:0 dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_9/dropout/MulИ
dropout_9/dropout/ShapeShape&sequential_3/dense_10/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape÷
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€# *
dtype020
.dropout_9/dropout/random_uniform/RandomUniformЙ
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_9/dropout/GreaterEqual/yк
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2 
dropout_9/dropout/GreaterEqual°
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€# 2
dropout_9/dropout/Cast¶
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_9/dropout/Mul_1Х
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
add_1ґ
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indicesб
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2$
"layer_normalization_7/moments/meanЋ
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2,
*layer_normalization_7/moments/StopGradientн
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 21
/layer_normalization_7/moments/SquaredDifferenceЊ
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indicesЧ
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2(
&layer_normalization_7/moments/varianceУ
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_7/batchnorm/add/yк
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2%
#layer_normalization_7/batchnorm/addґ
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€#2'
%layer_normalization_7/batchnorm/Rsqrtа
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOpо
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_7/batchnorm/mulњ
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/mul_1б
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/mul_2‘
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOpк
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_7/batchnorm/subб
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/add_1’
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp.^sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2М
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
:€€€€€€€€€# 
 
_user_specified_nameinputs
Љ
Д
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_301033
x'
#embedding_3_embedding_lookup_271001'
#embedding_2_embedding_lookup_271007
identityИҐembedding_2/embedding_lookupҐembedding_3/embedding_lookup?
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
rangeБ
embedding_3/embedding_lookupResourceGather#embedding_3_embedding_lookup_271001range:output:0*
Tindices0*6
_class,
*(loc:@embedding_3/embedding_lookup/271001*'
_output_shapes
:€€€€€€€€€ *
dtype02
embedding_3/embedding_lookupл
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_3/embedding_lookup/271001*'
_output_shapes
:€€€€€€€€€ 2'
%embedding_3/embedding_lookup/Identityј
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'embedding_3/embedding_lookup/Identity_1q
embedding_2/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€ДR2
embedding_2/CastМ
embedding_2/embedding_lookupResourceGather#embedding_2_embedding_lookup_271007embedding_2/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_2/embedding_lookup/271007*,
_output_shapes
:€€€€€€€€€ДR *
dtype02
embedding_2/embedding_lookupр
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_2/embedding_lookup/271007*,
_output_shapes
:€€€€€€€€€ДR 2'
%embedding_2/embedding_lookup/Identity≈
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2)
'embedding_2/embedding_lookup/Identity_1Ѓ
addAddV20embedding_2/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2
addЮ
IdentityIdentityadd:z:0^embedding_2/embedding_lookup^embedding_3/embedding_lookup*
T0*,
_output_shapes
:€€€€€€€€€ДR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€ДR::2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:K G
(
_output_shapes
:€€€€€€€€€ДR

_user_specified_namex
Љ0
»
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_309555

inputs
assignmovingavg_309530
assignmovingavg_1_309536)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/309530*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_309530*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/309530*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/309530*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_309530AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/309530*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/309536*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_309536*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/309536*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/309536*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_309536AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/309536*
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
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/add_1Ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
Щ
G
+__inference_dropout_10_layer_call_fn_311546

inputs
identity«
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3098242
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
В
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_311531

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
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
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
—
г
D__inference_dense_10_layer_call_and_return_conditional_losses_310521

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2	
BiasAddЬ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€#@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€#@
 
_user_specified_nameinputs
с	
Ё
D__inference_dense_11_layer_call_and_return_conditional_losses_311510

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	и@*
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
:€€€€€€€€€и::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
¶

’
)__inference_restored_function_body_306435

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
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_3028592
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
и
И
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_309575

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/add_1я
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
…
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_309824

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
Ц
И
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_309432

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1и
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
о
©
6__inference_batch_normalization_2_layer_call_fn_311299

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
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3092922
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
 
©
6__inference_batch_normalization_2_layer_call_fn_311217

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3095752
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
•
d
+__inference_dropout_11_layer_call_fn_311588

inputs
identityИҐStatefulPartitionedCallя
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3098762
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
¬
u
I__inference_concatenate_1_layer_call_and_return_conditional_losses_311493
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
:€€€€€€€€€и2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€а:€€€€€€€€€:R N
(
_output_shapes
:€€€€€€€€€а
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
ў№
÷
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_302342

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
identityИҐ.layer_normalization_6/batchnorm/ReadVariableOpҐ2layer_normalization_6/batchnorm/mul/ReadVariableOpҐ.layer_normalization_7/batchnorm/ReadVariableOpҐ2layer_normalization_7/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_3/attention_output/add/ReadVariableOpҐDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_3/key/add/ReadVariableOpҐ7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_3/query/add/ReadVariableOpҐ9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_3/value/add/ReadVariableOpҐ9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpҐ,sequential_3/dense_10/BiasAdd/ReadVariableOpҐ.sequential_3/dense_10/Tensordot/ReadVariableOpҐ+sequential_3/dense_9/BiasAdd/ReadVariableOpҐ-sequential_3/dense_9/Tensordot/ReadVariableOpэ
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/Einsumџ
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOpх
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2"
 multi_head_attention_3/query/addч
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/Einsum’
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOpн
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2 
multi_head_attention_3/key/addэ
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/Einsumџ
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOpх
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2"
 multi_head_attention_3/value/addБ
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_3/Mul/y∆
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€# 2
multi_head_attention_3/Mulь
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/Einsumƒ
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€##2(
&multi_head_attention_3/softmax/Softmax 
'multi_head_attention_3/dropout/IdentityIdentity0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€##2)
'multi_head_attention_3/dropout/IdentityФ
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/Identity:output:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/EinsumЮ
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsumш
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOpЭ
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2-
+multi_head_attention_3/attention_output/addЫ
dropout_8/IdentityIdentity/multi_head_attention_3/attention_output/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_8/Identityn
addAddV2inputsdropout_8/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
addґ
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indicesя
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2$
"layer_normalization_6/moments/meanЋ
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2,
*layer_normalization_6/moments/StopGradientл
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 21
/layer_normalization_6/moments/SquaredDifferenceЊ
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indicesЧ
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2(
&layer_normalization_6/moments/varianceУ
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_6/batchnorm/add/yк
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2%
#layer_normalization_6/batchnorm/addґ
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€#2'
%layer_normalization_6/batchnorm/Rsqrtа
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOpо
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_6/batchnorm/mulљ
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/mul_1б
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/mul_2‘
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOpк
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_6/batchnorm/subб
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/add_1’
-sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_3/dense_9/Tensordot/ReadVariableOpФ
#sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_3/dense_9/Tensordot/axesЫ
#sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_3/dense_9/Tensordot/free•
$sequential_3/dense_9/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/ShapeЮ
,sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/GatherV2/axisЇ
'sequential_3/dense_9/Tensordot/GatherV2GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/free:output:05sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/GatherV2Ґ
.sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_3/dense_9/Tensordot/GatherV2_1/axisј
)sequential_3/dense_9/Tensordot/GatherV2_1GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/axes:output:07sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_3/dense_9/Tensordot/GatherV2_1Ц
$sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_3/dense_9/Tensordot/Const‘
#sequential_3/dense_9/Tensordot/ProdProd0sequential_3/dense_9/Tensordot/GatherV2:output:0-sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_3/dense_9/Tensordot/ProdЪ
&sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_3/dense_9/Tensordot/Const_1№
%sequential_3/dense_9/Tensordot/Prod_1Prod2sequential_3/dense_9/Tensordot/GatherV2_1:output:0/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_3/dense_9/Tensordot/Prod_1Ъ
*sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_3/dense_9/Tensordot/concat/axisЩ
%sequential_3/dense_9/Tensordot/concatConcatV2,sequential_3/dense_9/Tensordot/free:output:0,sequential_3/dense_9/Tensordot/axes:output:03sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_9/Tensordot/concatа
$sequential_3/dense_9/Tensordot/stackPack,sequential_3/dense_9/Tensordot/Prod:output:0.sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/stackт
(sequential_3/dense_9/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0.sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2*
(sequential_3/dense_9/Tensordot/transposeу
&sequential_3/dense_9/Tensordot/ReshapeReshape,sequential_3/dense_9/Tensordot/transpose:y:0-sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_3/dense_9/Tensordot/Reshapeт
%sequential_3/dense_9/Tensordot/MatMulMatMul/sequential_3/dense_9/Tensordot/Reshape:output:05sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_3/dense_9/Tensordot/MatMulЪ
&sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_3/dense_9/Tensordot/Const_2Ю
,sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/concat_1/axis¶
'sequential_3/dense_9/Tensordot/concat_1ConcatV20sequential_3/dense_9/Tensordot/GatherV2:output:0/sequential_3/dense_9/Tensordot/Const_2:output:05sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/concat_1д
sequential_3/dense_9/TensordotReshape/sequential_3/dense_9/Tensordot/MatMul:product:00sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2 
sequential_3/dense_9/TensordotЋ
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOpџ
sequential_3/dense_9/BiasAddBiasAdd'sequential_3/dense_9/Tensordot:output:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
sequential_3/dense_9/BiasAddЫ
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
sequential_3/dense_9/ReluЎ
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOpЦ
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axesЭ
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free•
%sequential_3/dense_10/Tensordot/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape†
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axisњ
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2§
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axis≈
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1Ш
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/ConstЎ
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/ProdЬ
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1а
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1Ь
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axisЮ
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concatд
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stackу
)sequential_3/dense_10/Tensordot/transpose	Transpose'sequential_3/dense_9/Relu:activations:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2+
)sequential_3/dense_10/Tensordot/transposeч
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_3/dense_10/Tensordot/Reshapeц
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_3/dense_10/Tensordot/MatMulЬ
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_2†
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axisЂ
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1и
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2!
sequential_3/dense_10/Tensordotќ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOpя
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
sequential_3/dense_10/BiasAddТ
dropout_9/IdentityIdentity&sequential_3/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_9/IdentityХ
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
add_1ґ
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indicesб
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2$
"layer_normalization_7/moments/meanЋ
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2,
*layer_normalization_7/moments/StopGradientн
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 21
/layer_normalization_7/moments/SquaredDifferenceЊ
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indicesЧ
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2(
&layer_normalization_7/moments/varianceУ
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_7/batchnorm/add/yк
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2%
#layer_normalization_7/batchnorm/addґ
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€#2'
%layer_normalization_7/batchnorm/Rsqrtа
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOpо
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_7/batchnorm/mulњ
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/mul_1б
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/mul_2‘
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOpк
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_7/batchnorm/subб
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/add_1’
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp.^sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2М
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
:€€€€€€€€€# 
 
_user_specified_nameinputs
Ѓ
Ю
$__inference_signature_wrapper_310440
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

unknown_34
identityИҐStatefulPartitionedCall≤
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
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_3091182
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_3:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_4
Ј
k
A__inference_add_1_layer_call_and_return_conditional_losses_309708

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:€€€€€€€€€# 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:€€€€€€€€€# :€€€€€€€€€# :S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
п
~
)__inference_dense_10_layer_call_fn_311831

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
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_3105212
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€#@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€#@
 
_user_specified_nameinputs
¶

’
)__inference_restored_function_body_305739

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
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_3023422
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
о
©
6__inference_batch_normalization_3_layer_call_fn_311381

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
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3094322
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
и
И
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311437

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/add_1я
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
Т	
Ё
D__inference_dense_13_layer_call_and_return_conditional_losses_309904

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
¬І
Й
C__inference_model_1_layer_call_and_return_conditional_losses_310929
inputs_0
inputs_1)
%token_and_position_embedding_1_310795)
%token_and_position_embedding_1_3107978
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
9batch_normalization_3_batchnorm_readvariableop_2_resource
transformer_block_3_310869
transformer_block_3_310871
transformer_block_3_310873
transformer_block_3_310875
transformer_block_3_310877
transformer_block_3_310879
transformer_block_3_310881
transformer_block_3_310883
transformer_block_3_310885
transformer_block_3_310887
transformer_block_3_310889
transformer_block_3_310891
transformer_block_3_310893
transformer_block_3_310895
transformer_block_3_310897
transformer_block_3_310899+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource
identityИҐ.batch_normalization_2/batchnorm/ReadVariableOpҐ0batch_normalization_2/batchnorm/ReadVariableOp_1Ґ0batch_normalization_2/batchnorm/ReadVariableOp_2Ґ2batch_normalization_2/batchnorm/mul/ReadVariableOpҐ.batch_normalization_3/batchnorm/ReadVariableOpҐ0batch_normalization_3/batchnorm/ReadVariableOp_1Ґ0batch_normalization_3/batchnorm/ReadVariableOp_2Ґ2batch_normalization_3/batchnorm/mul/ReadVariableOpҐconv1d_2/BiasAdd/ReadVariableOpҐ+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpҐconv1d_3/BiasAdd/ReadVariableOpҐ+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpҐdense_11/BiasAdd/ReadVariableOpҐdense_11/MatMul/ReadVariableOpҐdense_12/BiasAdd/ReadVariableOpҐdense_12/MatMul/ReadVariableOpҐdense_13/BiasAdd/ReadVariableOpҐdense_13/MatMul/ReadVariableOpҐ6token_and_position_embedding_1/StatefulPartitionedCallҐ+transformer_block_3/StatefulPartitionedCallџ
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_0%token_and_position_embedding_1_310795%token_and_position_embedding_1_310797*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_30562828
6token_and_position_embedding_1/StatefulPartitionedCallЛ
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2 
conv1d_2/conv1d/ExpandDims/dimл
conv1d_2/conv1d/ExpandDims
ExpandDims?token_and_position_embedding_1/StatefulPartitionedCall:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2
conv1d_2/conv1d/ExpandDims”
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimџ
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_2/conv1d/ExpandDims_1џ
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR *
paddingSAME*
strides
2
conv1d_2/conv1dЃ
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR *
squeeze_dims

э€€€€€€€€2
conv1d_2/conv1d/SqueezeІ
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_2/BiasAdd/ReadVariableOp±
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2
conv1d_2/BiasAddx
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2
conv1d_2/ReluК
"average_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_3/ExpandDims/dim”
average_pooling1d_3/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0+average_pooling1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2 
average_pooling1d_3/ExpandDimsе
average_pooling1d_3/AvgPoolAvgPool'average_pooling1d_3/ExpandDims:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё *
ksize
*
paddingVALID*
strides
2
average_pooling1d_3/AvgPoolє
average_pooling1d_3/SqueezeSqueeze$average_pooling1d_3/AvgPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё *
squeeze_dims
2
average_pooling1d_3/SqueezeЛ
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2 
conv1d_3/conv1d/ExpandDims/dim–
conv1d_3/conv1d/ExpandDims
ExpandDims$average_pooling1d_3/Squeeze:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё 2
conv1d_3/conv1d/ExpandDims”
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimџ
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_3/conv1d/ExpandDims_1џ
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё *
paddingSAME*
strides
2
conv1d_3/conv1dЃ
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё *
squeeze_dims

э€€€€€€€€2
conv1d_3/conv1d/SqueezeІ
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_3/BiasAdd/ReadVariableOp±
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ё 2
conv1d_3/BiasAddx
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё 2
conv1d_3/ReluК
"average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_5/ExpandDims/dimч
average_pooling1d_5/ExpandDims
ExpandDims?token_and_position_embedding_1/StatefulPartitionedCall:output:0+average_pooling1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2 
average_pooling1d_5/ExpandDimsж
average_pooling1d_5/AvgPoolAvgPool'average_pooling1d_5/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€# *
ksize	
ђ*
paddingVALID*
strides	
ђ2
average_pooling1d_5/AvgPoolЄ
average_pooling1d_5/SqueezeSqueeze$average_pooling1d_5/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€# *
squeeze_dims
2
average_pooling1d_5/SqueezeК
"average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_4/ExpandDims/dim”
average_pooling1d_4/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0+average_pooling1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё 2 
average_pooling1d_4/ExpandDimsд
average_pooling1d_4/AvgPoolAvgPool'average_pooling1d_4/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€# *
ksize

*
paddingVALID*
strides

2
average_pooling1d_4/AvgPoolЄ
average_pooling1d_4/SqueezeSqueeze$average_pooling1d_4/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€# *
squeeze_dims
2
average_pooling1d_4/Squeeze‘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yа
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/add•
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_2/batchnorm/Rsqrtа
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/mulЏ
%batch_normalization_2/batchnorm/mul_1Mul$average_pooling1d_4/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%batch_normalization_2/batchnorm/mul_1Џ
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1Ё
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_2/batchnorm/mul_2Џ
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2џ
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_2/batchnorm/subб
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%batch_normalization_2/batchnorm/add_1‘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yа
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/add•
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/Rsqrtа
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/mulЏ
%batch_normalization_3/batchnorm/mul_1Mul$average_pooling1d_5/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%batch_normalization_3/batchnorm/mul_1Џ
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1Ё
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/mul_2Џ
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2џ
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/subб
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%batch_normalization_3/batchnorm/add_1Ђ
	add_1/addAddV2)batch_normalization_2/batchnorm/add_1:z:0)batch_normalization_3/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
	add_1/add„
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/add:z:0transformer_block_3_310869transformer_block_3_310871transformer_block_3_310873transformer_block_3_310875transformer_block_3_310877transformer_block_3_310879transformer_block_3_310881transformer_block_3_310883transformer_block_3_310885transformer_block_3_310887transformer_block_3_310889transformer_block_3_310891transformer_block_3_310893transformer_block_3_310895transformer_block_3_310897transformer_block_3_310899*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_3057392-
+transformer_block_3/StatefulPartitionedCalls
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€`  2
flatten_1/Constі
flatten_1/ReshapeReshape4transformer_block_3/StatefulPartitionedCall:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€а2
flatten_1/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisЊ
concatenate_1/concatConcatV2flatten_1/Reshape:output:0inputs_1"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€и2
concatenate_1/concat©
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	и@*
dtype02 
dense_11/MatMul/ReadVariableOp•
dense_11/MatMulMatMulconcatenate_1/concat:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_11/MatMulІ
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_11/BiasAdd/ReadVariableOp•
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_11/BiasAdds
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_11/ReluЕ
dropout_10/IdentityIdentitydense_11/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_10/Identity®
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_12/MatMul/ReadVariableOp§
dense_12/MatMulMatMuldropout_10/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/MatMulІ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp•
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_12/ReluЕ
dropout_11/IdentityIdentitydense_12/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_11/Identity®
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_13/MatMul/ReadVariableOp§
dense_13/MatMulMatMuldropout_11/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_13/MatMulІ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp•
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_13/BiasAdd’
IdentityIdentitydense_13/BiasAdd:output:0/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::2`
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
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:R N
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
Т	
Ё
D__inference_dense_13_layer_call_and_return_conditional_losses_311603

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
Ц
И
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311273

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1и
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
у0
»
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_309399

inputs
assignmovingavg_309374
assignmovingavg_1_309380)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/309374*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_309374*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/309374*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/309374*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_309374AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/309374*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/309380*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_309380*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/309380*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/309380*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_309380AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/309380*
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
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
у
~
)__inference_conv1d_2_layer_call_fn_311110

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3094692
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€ДR 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€ДR ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ДR 
 
_user_specified_nameinputs
Є
†
-__inference_sequential_3_layer_call_fn_311739

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
 *+
_output_shapes
:€€€€€€€€€# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3105692
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
…
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_309881

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
–

а
4__inference_transformer_block_3_layer_call_fn_303075

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
 *+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_3030542
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
Д
P
4__inference_average_pooling1d_3_layer_call_fn_309133

inputs
identityж
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
GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_3091272
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
В
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_309876

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
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
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
С[
и
C__inference_model_1_layer_call_and_return_conditional_losses_310113

inputs
inputs_1)
%token_and_position_embedding_1_310023)
%token_and_position_embedding_1_310025
conv1d_2_310028
conv1d_2_310030
conv1d_3_310034
conv1d_3_310036 
batch_normalization_2_310041 
batch_normalization_2_310043 
batch_normalization_2_310045 
batch_normalization_2_310047 
batch_normalization_3_310050 
batch_normalization_3_310052 
batch_normalization_3_310054 
batch_normalization_3_310056
transformer_block_3_310060
transformer_block_3_310062
transformer_block_3_310064
transformer_block_3_310066
transformer_block_3_310068
transformer_block_3_310070
transformer_block_3_310072
transformer_block_3_310074
transformer_block_3_310076
transformer_block_3_310078
transformer_block_3_310080
transformer_block_3_310082
transformer_block_3_310084
transformer_block_3_310086
transformer_block_3_310088
transformer_block_3_310090
dense_11_310095
dense_11_310097
dense_12_310101
dense_12_310103
dense_13_310107
dense_13_310109
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ conv1d_2/StatefulPartitionedCallҐ conv1d_3/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ"dropout_10/StatefulPartitionedCallҐ"dropout_11/StatefulPartitionedCallҐ6token_and_position_embedding_1/StatefulPartitionedCallҐ+transformer_block_3/StatefulPartitionedCallў
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_1_310023%token_and_position_embedding_1_310025*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_30562828
6token_and_position_embedding_1/StatefulPartitionedCall’
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0conv1d_2_310028conv1d_2_310030*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3094692"
 conv1d_2/StatefulPartitionedCall†
#average_pooling1d_3/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ё * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_3091272%
#average_pooling1d_3/PartitionedCall¬
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_3/PartitionedCall:output:0conv1d_3_310034conv1d_3_310036*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ё *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_3095022"
 conv1d_3/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_3091572%
#average_pooling1d_5/PartitionedCallЯ
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_3091422%
#average_pooling1d_4/PartitionedCallј
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_2_310041batch_normalization_2_310043batch_normalization_2_310045batch_normalization_2_310047*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3095552/
-batch_normalization_2/StatefulPartitionedCallј
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_3_310050batch_normalization_3_310052batch_normalization_3_310054batch_normalization_3_310056*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3096462/
-batch_normalization_3/StatefulPartitionedCallї
add_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:06batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3097082
add_1/PartitionedCallи
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0transformer_block_3_310060transformer_block_3_310062transformer_block_3_310064transformer_block_3_310066transformer_block_3_310068transformer_block_3_310070transformer_block_3_310072transformer_block_3_310074transformer_block_3_310076transformer_block_3_310078transformer_block_3_310080transformer_block_3_310082transformer_block_3_310084transformer_block_3_310086transformer_block_3_310088transformer_block_3_310090*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_3064352-
+transformer_block_3/StatefulPartitionedCallЙ
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€а* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_3097562
flatten_1/PartitionedCallО
concatenate_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_3097712
concatenate_1/PartitionedCallЈ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_11_310095dense_11_310097*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_3097912"
 dense_11/StatefulPartitionedCallШ
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3098192$
"dropout_10/StatefulPartitionedCallЉ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_12_310101dense_12_310103*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3098482"
 dense_12/StatefulPartitionedCallљ
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3098762$
"dropout_11/StatefulPartitionedCallЉ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_13_310107dense_13_310109*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_3099042"
 dense_13/StatefulPartitionedCallљ
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ДR
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
б
~
)__inference_dense_11_layer_call_fn_311519

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallч
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
GPU2*0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_3097912
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€и::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
В
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_311578

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
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
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
У[
и
C__inference_model_1_layer_call_and_return_conditional_losses_309921
input_3
input_4)
%token_and_position_embedding_1_309448)
%token_and_position_embedding_1_309450
conv1d_2_309480
conv1d_2_309482
conv1d_3_309513
conv1d_3_309515 
batch_normalization_2_309602 
batch_normalization_2_309604 
batch_normalization_2_309606 
batch_normalization_2_309608 
batch_normalization_3_309693 
batch_normalization_3_309695 
batch_normalization_3_309697 
batch_normalization_3_309699
transformer_block_3_309717
transformer_block_3_309719
transformer_block_3_309721
transformer_block_3_309723
transformer_block_3_309725
transformer_block_3_309727
transformer_block_3_309729
transformer_block_3_309731
transformer_block_3_309733
transformer_block_3_309735
transformer_block_3_309737
transformer_block_3_309739
transformer_block_3_309741
transformer_block_3_309743
transformer_block_3_309745
transformer_block_3_309747
dense_11_309802
dense_11_309804
dense_12_309859
dense_12_309861
dense_13_309915
dense_13_309917
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ conv1d_2/StatefulPartitionedCallҐ conv1d_3/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ"dropout_10/StatefulPartitionedCallҐ"dropout_11/StatefulPartitionedCallҐ6token_and_position_embedding_1/StatefulPartitionedCallҐ+transformer_block_3/StatefulPartitionedCallЏ
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_3%token_and_position_embedding_1_309448%token_and_position_embedding_1_309450*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_30562828
6token_and_position_embedding_1/StatefulPartitionedCall’
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0conv1d_2_309480conv1d_2_309482*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3094692"
 conv1d_2/StatefulPartitionedCall†
#average_pooling1d_3/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ё * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_3091272%
#average_pooling1d_3/PartitionedCall¬
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_3/PartitionedCall:output:0conv1d_3_309513conv1d_3_309515*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ё *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_3095022"
 conv1d_3/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_3091572%
#average_pooling1d_5/PartitionedCallЯ
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_3091422%
#average_pooling1d_4/PartitionedCallј
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_2_309602batch_normalization_2_309604batch_normalization_2_309606batch_normalization_2_309608*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3095552/
-batch_normalization_2/StatefulPartitionedCallј
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_3_309693batch_normalization_3_309695batch_normalization_3_309697batch_normalization_3_309699*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3096462/
-batch_normalization_3/StatefulPartitionedCallї
add_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:06batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3097082
add_1/PartitionedCallи
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0transformer_block_3_309717transformer_block_3_309719transformer_block_3_309721transformer_block_3_309723transformer_block_3_309725transformer_block_3_309727transformer_block_3_309729transformer_block_3_309731transformer_block_3_309733transformer_block_3_309735transformer_block_3_309737transformer_block_3_309739transformer_block_3_309741transformer_block_3_309743transformer_block_3_309745transformer_block_3_309747*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_3064352-
+transformer_block_3/StatefulPartitionedCallЙ
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€а* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_3097562
flatten_1/PartitionedCallН
concatenate_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0input_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_3097712
concatenate_1/PartitionedCallЈ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_11_309802dense_11_309804*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_3097912"
 dense_11/StatefulPartitionedCallШ
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3098192$
"dropout_10/StatefulPartitionedCallЉ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_12_309859dense_12_309861*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3098482"
 dense_12/StatefulPartitionedCallљ
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3098762$
"dropout_11/StatefulPartitionedCallЉ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_13_309915dense_13_309917*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_3099042"
 dense_13/StatefulPartitionedCallљ
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_3:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_4
о	
Ё
D__inference_dense_12_layer_call_and_return_conditional_losses_311557

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
–
Ґ
(__inference_model_1_layer_call_fn_310188
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

unknown_34
identityИҐStatefulPartitionedCall–
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
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_3101132
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_3:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_4
‘
Ґ
(__inference_model_1_layer_call_fn_310360
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

unknown_34
identityИҐStatefulPartitionedCall‘
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
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_3102852
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_3:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_4
х
k
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_309127

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

ExpandDimsЇ
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
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
Ї
s
I__inference_concatenate_1_layer_call_and_return_conditional_losses_309771

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
:€€€€€€€€€и2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€и2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€а:€€€€€€€€€:P L
(
_output_shapes
:€€€€€€€€€а
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м
©
6__inference_batch_normalization_3_layer_call_fn_311368

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
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3093992
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
бњ
•
!__inference__wrapped_model_309118
input_3
input_41
-model_1_token_and_position_embedding_1_3089841
-model_1_token_and_position_embedding_1_308986@
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
Amodel_1_batch_normalization_3_batchnorm_readvariableop_2_resource&
"model_1_transformer_block_3_309058&
"model_1_transformer_block_3_309060&
"model_1_transformer_block_3_309062&
"model_1_transformer_block_3_309064&
"model_1_transformer_block_3_309066&
"model_1_transformer_block_3_309068&
"model_1_transformer_block_3_309070&
"model_1_transformer_block_3_309072&
"model_1_transformer_block_3_309074&
"model_1_transformer_block_3_309076&
"model_1_transformer_block_3_309078&
"model_1_transformer_block_3_309080&
"model_1_transformer_block_3_309082&
"model_1_transformer_block_3_309084&
"model_1_transformer_block_3_309086&
"model_1_transformer_block_3_3090883
/model_1_dense_11_matmul_readvariableop_resource4
0model_1_dense_11_biasadd_readvariableop_resource3
/model_1_dense_12_matmul_readvariableop_resource4
0model_1_dense_12_biasadd_readvariableop_resource3
/model_1_dense_13_matmul_readvariableop_resource4
0model_1_dense_13_biasadd_readvariableop_resource
identityИҐ6model_1/batch_normalization_2/batchnorm/ReadVariableOpҐ8model_1/batch_normalization_2/batchnorm/ReadVariableOp_1Ґ8model_1/batch_normalization_2/batchnorm/ReadVariableOp_2Ґ:model_1/batch_normalization_2/batchnorm/mul/ReadVariableOpҐ6model_1/batch_normalization_3/batchnorm/ReadVariableOpҐ8model_1/batch_normalization_3/batchnorm/ReadVariableOp_1Ґ8model_1/batch_normalization_3/batchnorm/ReadVariableOp_2Ґ:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOpҐ'model_1/conv1d_2/BiasAdd/ReadVariableOpҐ3model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpҐ'model_1/conv1d_3/BiasAdd/ReadVariableOpҐ3model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpҐ'model_1/dense_11/BiasAdd/ReadVariableOpҐ&model_1/dense_11/MatMul/ReadVariableOpҐ'model_1/dense_12/BiasAdd/ReadVariableOpҐ&model_1/dense_12/MatMul/ReadVariableOpҐ'model_1/dense_13/BiasAdd/ReadVariableOpҐ&model_1/dense_13/MatMul/ReadVariableOpҐ>model_1/token_and_position_embedding_1/StatefulPartitionedCallҐ3model_1/transformer_block_3/StatefulPartitionedCallъ
>model_1/token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_3-model_1_token_and_position_embedding_1_308984-model_1_token_and_position_embedding_1_308986*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_3056282@
>model_1/token_and_position_embedding_1/StatefulPartitionedCallЫ
&model_1/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2(
&model_1/conv1d_2/conv1d/ExpandDims/dimЛ
"model_1/conv1d_2/conv1d/ExpandDims
ExpandDimsGmodel_1/token_and_position_embedding_1/StatefulPartitionedCall:output:0/model_1/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2$
"model_1/conv1d_2/conv1d/ExpandDimsл
3model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_1_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype025
3model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЦ
(model_1/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_1/conv1d_2/conv1d/ExpandDims_1/dimы
$model_1/conv1d_2/conv1d/ExpandDims_1
ExpandDims;model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:01model_1/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2&
$model_1/conv1d_2/conv1d/ExpandDims_1ы
model_1/conv1d_2/conv1dConv2D+model_1/conv1d_2/conv1d/ExpandDims:output:0-model_1/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR *
paddingSAME*
strides
2
model_1/conv1d_2/conv1d∆
model_1/conv1d_2/conv1d/SqueezeSqueeze model_1/conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR *
squeeze_dims

э€€€€€€€€2!
model_1/conv1d_2/conv1d/Squeezeњ
'model_1/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv1d_2/BiasAdd/ReadVariableOp—
model_1/conv1d_2/BiasAddBiasAdd(model_1/conv1d_2/conv1d/Squeeze:output:0/model_1/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2
model_1/conv1d_2/BiasAddР
model_1/conv1d_2/ReluRelu!model_1/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2
model_1/conv1d_2/ReluЪ
*model_1/average_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_1/average_pooling1d_3/ExpandDims/dimу
&model_1/average_pooling1d_3/ExpandDims
ExpandDims#model_1/conv1d_2/Relu:activations:03model_1/average_pooling1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2(
&model_1/average_pooling1d_3/ExpandDimsэ
#model_1/average_pooling1d_3/AvgPoolAvgPool/model_1/average_pooling1d_3/ExpandDims:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё *
ksize
*
paddingVALID*
strides
2%
#model_1/average_pooling1d_3/AvgPool—
#model_1/average_pooling1d_3/SqueezeSqueeze,model_1/average_pooling1d_3/AvgPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё *
squeeze_dims
2%
#model_1/average_pooling1d_3/SqueezeЫ
&model_1/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2(
&model_1/conv1d_3/conv1d/ExpandDims/dimр
"model_1/conv1d_3/conv1d/ExpandDims
ExpandDims,model_1/average_pooling1d_3/Squeeze:output:0/model_1/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё 2$
"model_1/conv1d_3/conv1d/ExpandDimsл
3model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_1_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype025
3model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpЦ
(model_1/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_1/conv1d_3/conv1d/ExpandDims_1/dimы
$model_1/conv1d_3/conv1d/ExpandDims_1
ExpandDims;model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:01model_1/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2&
$model_1/conv1d_3/conv1d/ExpandDims_1ы
model_1/conv1d_3/conv1dConv2D+model_1/conv1d_3/conv1d/ExpandDims:output:0-model_1/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё *
paddingSAME*
strides
2
model_1/conv1d_3/conv1d∆
model_1/conv1d_3/conv1d/SqueezeSqueeze model_1/conv1d_3/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё *
squeeze_dims

э€€€€€€€€2!
model_1/conv1d_3/conv1d/Squeezeњ
'model_1/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv1d_3/BiasAdd/ReadVariableOp—
model_1/conv1d_3/BiasAddBiasAdd(model_1/conv1d_3/conv1d/Squeeze:output:0/model_1/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ё 2
model_1/conv1d_3/BiasAddР
model_1/conv1d_3/ReluRelu!model_1/conv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё 2
model_1/conv1d_3/ReluЪ
*model_1/average_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_1/average_pooling1d_5/ExpandDims/dimЧ
&model_1/average_pooling1d_5/ExpandDims
ExpandDimsGmodel_1/token_and_position_embedding_1/StatefulPartitionedCall:output:03model_1/average_pooling1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ДR 2(
&model_1/average_pooling1d_5/ExpandDimsю
#model_1/average_pooling1d_5/AvgPoolAvgPool/model_1/average_pooling1d_5/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€# *
ksize	
ђ*
paddingVALID*
strides	
ђ2%
#model_1/average_pooling1d_5/AvgPool–
#model_1/average_pooling1d_5/SqueezeSqueeze,model_1/average_pooling1d_5/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€# *
squeeze_dims
2%
#model_1/average_pooling1d_5/SqueezeЪ
*model_1/average_pooling1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_1/average_pooling1d_4/ExpandDims/dimу
&model_1/average_pooling1d_4/ExpandDims
ExpandDims#model_1/conv1d_3/Relu:activations:03model_1/average_pooling1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё 2(
&model_1/average_pooling1d_4/ExpandDimsь
#model_1/average_pooling1d_4/AvgPoolAvgPool/model_1/average_pooling1d_4/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€# *
ksize

*
paddingVALID*
strides

2%
#model_1/average_pooling1d_4/AvgPool–
#model_1/average_pooling1d_4/SqueezeSqueeze,model_1/average_pooling1d_4/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€# *
squeeze_dims
2%
#model_1/average_pooling1d_4/Squeezeм
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
 *oГ:2/
-model_1/batch_normalization_2/batchnorm/add/yА
+model_1/batch_normalization_2/batchnorm/addAddV2>model_1/batch_normalization_2/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_2/batchnorm/addљ
-model_1/batch_normalization_2/batchnorm/RsqrtRsqrt/model_1/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_1/batch_normalization_2/batchnorm/Rsqrtш
:model_1/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_1/batch_normalization_2/batchnorm/mul/ReadVariableOpэ
+model_1/batch_normalization_2/batchnorm/mulMul1model_1/batch_normalization_2/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_2/batchnorm/mulъ
-model_1/batch_normalization_2/batchnorm/mul_1Mul,model_1/average_pooling1d_4/Squeeze:output:0/model_1/batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2/
-model_1/batch_normalization_2/batchnorm/mul_1т
8model_1/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_1/batch_normalization_2/batchnorm/ReadVariableOp_1э
-model_1/batch_normalization_2/batchnorm/mul_2Mul@model_1/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_1/batch_normalization_2/batchnorm/mul_2т
8model_1/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_1/batch_normalization_2/batchnorm/ReadVariableOp_2ы
+model_1/batch_normalization_2/batchnorm/subSub@model_1/batch_normalization_2/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_2/batchnorm/subБ
-model_1/batch_normalization_2/batchnorm/add_1AddV21model_1/batch_normalization_2/batchnorm/mul_1:z:0/model_1/batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2/
-model_1/batch_normalization_2/batchnorm/add_1м
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
 *oГ:2/
-model_1/batch_normalization_3/batchnorm/add/yА
+model_1/batch_normalization_3/batchnorm/addAddV2>model_1/batch_normalization_3/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_3/batchnorm/addљ
-model_1/batch_normalization_3/batchnorm/RsqrtRsqrt/model_1/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_1/batch_normalization_3/batchnorm/Rsqrtш
:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_1/batch_normalization_3/batchnorm/mul/ReadVariableOpэ
+model_1/batch_normalization_3/batchnorm/mulMul1model_1/batch_normalization_3/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_3/batchnorm/mulъ
-model_1/batch_normalization_3/batchnorm/mul_1Mul,model_1/average_pooling1d_5/Squeeze:output:0/model_1/batch_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2/
-model_1/batch_normalization_3/batchnorm/mul_1т
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_1э
-model_1/batch_normalization_3/batchnorm/mul_2Mul@model_1/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_1/batch_normalization_3/batchnorm/mul_2т
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_1/batch_normalization_3/batchnorm/ReadVariableOp_2ы
+model_1/batch_normalization_3/batchnorm/subSub@model_1/batch_normalization_3/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_1/batch_normalization_3/batchnorm/subБ
-model_1/batch_normalization_3/batchnorm/add_1AddV21model_1/batch_normalization_3/batchnorm/mul_1:z:0/model_1/batch_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2/
-model_1/batch_normalization_3/batchnorm/add_1Ћ
model_1/add_1/addAddV21model_1/batch_normalization_2/batchnorm/add_1:z:01model_1/batch_normalization_3/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
model_1/add_1/addп
3model_1/transformer_block_3/StatefulPartitionedCallStatefulPartitionedCallmodel_1/add_1/add:z:0"model_1_transformer_block_3_309058"model_1_transformer_block_3_309060"model_1_transformer_block_3_309062"model_1_transformer_block_3_309064"model_1_transformer_block_3_309066"model_1_transformer_block_3_309068"model_1_transformer_block_3_309070"model_1_transformer_block_3_309072"model_1_transformer_block_3_309074"model_1_transformer_block_3_309076"model_1_transformer_block_3_309078"model_1_transformer_block_3_309080"model_1_transformer_block_3_309082"model_1_transformer_block_3_309084"model_1_transformer_block_3_309086"model_1_transformer_block_3_309088*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_30573925
3model_1/transformer_block_3/StatefulPartitionedCallГ
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€`  2
model_1/flatten_1/Const‘
model_1/flatten_1/ReshapeReshape<model_1/transformer_block_3/StatefulPartitionedCall:output:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€а2
model_1/flatten_1/ReshapeИ
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_1/concat/axisЁ
model_1/concatenate_1/concatConcatV2"model_1/flatten_1/Reshape:output:0input_4*model_1/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€и2
model_1/concatenate_1/concatЅ
&model_1/dense_11/MatMul/ReadVariableOpReadVariableOp/model_1_dense_11_matmul_readvariableop_resource*
_output_shapes
:	и@*
dtype02(
&model_1/dense_11/MatMul/ReadVariableOp≈
model_1/dense_11/MatMulMatMul%model_1/concatenate_1/concat:output:0.model_1/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dense_11/MatMulњ
'model_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/dense_11/BiasAdd/ReadVariableOp≈
model_1/dense_11/BiasAddBiasAdd!model_1/dense_11/MatMul:product:0/model_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dense_11/BiasAddЛ
model_1/dense_11/ReluRelu!model_1/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dense_11/ReluЭ
model_1/dropout_10/IdentityIdentity#model_1/dense_11/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dropout_10/Identityј
&model_1/dense_12/MatMul/ReadVariableOpReadVariableOp/model_1_dense_12_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_1/dense_12/MatMul/ReadVariableOpƒ
model_1/dense_12/MatMulMatMul$model_1/dropout_10/Identity:output:0.model_1/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dense_12/MatMulњ
'model_1/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/dense_12/BiasAdd/ReadVariableOp≈
model_1/dense_12/BiasAddBiasAdd!model_1/dense_12/MatMul:product:0/model_1/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dense_12/BiasAddЛ
model_1/dense_12/ReluRelu!model_1/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dense_12/ReluЭ
model_1/dropout_11/IdentityIdentity#model_1/dense_12/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dropout_11/Identityј
&model_1/dense_13/MatMul/ReadVariableOpReadVariableOp/model_1_dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_1/dense_13/MatMul/ReadVariableOpƒ
model_1/dense_13/MatMulMatMul$model_1/dropout_11/Identity:output:0.model_1/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_1/dense_13/MatMulњ
'model_1/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_13/BiasAdd/ReadVariableOp≈
model_1/dense_13/BiasAddBiasAdd!model_1/dense_13/MatMul:product:0/model_1/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_1/dense_13/BiasAddэ
IdentityIdentity!model_1/dense_13/BiasAdd:output:07^model_1/batch_normalization_2/batchnorm/ReadVariableOp9^model_1/batch_normalization_2/batchnorm/ReadVariableOp_19^model_1/batch_normalization_2/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_2/batchnorm/mul/ReadVariableOp7^model_1/batch_normalization_3/batchnorm/ReadVariableOp9^model_1/batch_normalization_3/batchnorm/ReadVariableOp_19^model_1/batch_normalization_3/batchnorm/ReadVariableOp_2;^model_1/batch_normalization_3/batchnorm/mul/ReadVariableOp(^model_1/conv1d_2/BiasAdd/ReadVariableOp4^model_1/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp(^model_1/conv1d_3/BiasAdd/ReadVariableOp4^model_1/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp(^model_1/dense_11/BiasAdd/ReadVariableOp'^model_1/dense_11/MatMul/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp?^model_1/token_and_position_embedding_1/StatefulPartitionedCall4^model_1/transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::2p
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
'model_1/dense_11/BiasAdd/ReadVariableOp'model_1/dense_11/BiasAdd/ReadVariableOp2P
&model_1/dense_11/MatMul/ReadVariableOp&model_1/dense_11/MatMul/ReadVariableOp2R
'model_1/dense_12/BiasAdd/ReadVariableOp'model_1/dense_12/BiasAdd/ReadVariableOp2P
&model_1/dense_12/MatMul/ReadVariableOp&model_1/dense_12/MatMul/ReadVariableOp2R
'model_1/dense_13/BiasAdd/ReadVariableOp'model_1/dense_13/BiasAdd/ReadVariableOp2P
&model_1/dense_13/MatMul/ReadVariableOp&model_1/dense_13/MatMul/ReadVariableOp2А
>model_1/token_and_position_embedding_1/StatefulPartitionedCall>model_1/token_and_position_embedding_1/StatefulPartitionedCall2j
3model_1/transformer_block_3/StatefulPartitionedCall3model_1/transformer_block_3/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_3:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_4
у
~
)__inference_conv1d_3_layer_call_fn_311135

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ё *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_3095022
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€ё 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€ё ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€ё 
 
_user_specified_nameinputs
х
k
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_309142

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

ExpandDimsЇ
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize

*
paddingVALID*
strides

2	
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
я
~
)__inference_dense_12_layer_call_fn_311566

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallч
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
GPU2*0J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3098482
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
КX
Ю
C__inference_model_1_layer_call_and_return_conditional_losses_310015
input_3
input_4)
%token_and_position_embedding_1_309925)
%token_and_position_embedding_1_309927
conv1d_2_309930
conv1d_2_309932
conv1d_3_309936
conv1d_3_309938 
batch_normalization_2_309943 
batch_normalization_2_309945 
batch_normalization_2_309947 
batch_normalization_2_309949 
batch_normalization_3_309952 
batch_normalization_3_309954 
batch_normalization_3_309956 
batch_normalization_3_309958
transformer_block_3_309962
transformer_block_3_309964
transformer_block_3_309966
transformer_block_3_309968
transformer_block_3_309970
transformer_block_3_309972
transformer_block_3_309974
transformer_block_3_309976
transformer_block_3_309978
transformer_block_3_309980
transformer_block_3_309982
transformer_block_3_309984
transformer_block_3_309986
transformer_block_3_309988
transformer_block_3_309990
transformer_block_3_309992
dense_11_309997
dense_11_309999
dense_12_310003
dense_12_310005
dense_13_310009
dense_13_310011
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ conv1d_2/StatefulPartitionedCallҐ conv1d_3/StatefulPartitionedCallҐ dense_11/StatefulPartitionedCallҐ dense_12/StatefulPartitionedCallҐ dense_13/StatefulPartitionedCallҐ6token_and_position_embedding_1/StatefulPartitionedCallҐ+transformer_block_3/StatefulPartitionedCallЏ
6token_and_position_embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_3%token_and_position_embedding_1_309925%token_and_position_embedding_1_309927*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_30562828
6token_and_position_embedding_1/StatefulPartitionedCall’
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0conv1d_2_309930conv1d_2_309932*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_3094692"
 conv1d_2/StatefulPartitionedCall†
#average_pooling1d_3/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ё * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_3091272%
#average_pooling1d_3/PartitionedCall¬
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_3/PartitionedCall:output:0conv1d_3_309936conv1d_3_309938*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€ё *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_3095022"
 conv1d_3/StatefulPartitionedCallµ
#average_pooling1d_5/PartitionedCallPartitionedCall?token_and_position_embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_3091572%
#average_pooling1d_5/PartitionedCallЯ
#average_pooling1d_4/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_3091422%
#average_pooling1d_4/PartitionedCall¬
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_4/PartitionedCall:output:0batch_normalization_2_309943batch_normalization_2_309945batch_normalization_2_309947batch_normalization_2_309949*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3095752/
-batch_normalization_2/StatefulPartitionedCall¬
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_5/PartitionedCall:output:0batch_normalization_3_309952batch_normalization_3_309954batch_normalization_3_309956batch_normalization_3_309958*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3096662/
-batch_normalization_3/StatefulPartitionedCallї
add_1/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:06batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3097082
add_1/PartitionedCallи
+transformer_block_3/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0transformer_block_3_309962transformer_block_3_309964transformer_block_3_309966transformer_block_3_309968transformer_block_3_309970transformer_block_3_309972transformer_block_3_309974transformer_block_3_309976transformer_block_3_309978transformer_block_3_309980transformer_block_3_309982transformer_block_3_309984transformer_block_3_309986transformer_block_3_309988transformer_block_3_309990transformer_block_3_309992*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_3057392-
+transformer_block_3/StatefulPartitionedCallЙ
flatten_1/PartitionedCallPartitionedCall4transformer_block_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€а* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_3097562
flatten_1/PartitionedCallН
concatenate_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0input_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_3097712
concatenate_1/PartitionedCallЈ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_11_309997dense_11_309999*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_3097912"
 dense_11/StatefulPartitionedCallА
dropout_10/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3098242
dropout_10/PartitionedCallі
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_12_310003dense_12_310005*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_3098482"
 dense_12/StatefulPartitionedCallА
dropout_11/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3098812
dropout_11/PartitionedCallі
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_13_310009dense_13_310011*
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
GPU2*0J 8В *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_3099042"
 dense_13/StatefulPartitionedCallу
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall7^token_and_position_embedding_1/StatefulPartitionedCall,^transformer_block_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*ћ
_input_shapesЇ
Ј:€€€€€€€€€ДR:€€€€€€€€€::::::::::::::::::::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2p
6token_and_position_embedding_1/StatefulPartitionedCall6token_and_position_embedding_1/StatefulPartitionedCall2Z
+transformer_block_3/StatefulPartitionedCall+transformer_block_3/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€ДR
!
_user_specified_name	input_3:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_4
Љ
Д
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_303097
x'
#embedding_3_embedding_lookup_268908'
#embedding_2_embedding_lookup_268914
identityИҐembedding_2/embedding_lookupҐembedding_3/embedding_lookup?
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
rangeБ
embedding_3/embedding_lookupResourceGather#embedding_3_embedding_lookup_268908range:output:0*
Tindices0*6
_class,
*(loc:@embedding_3/embedding_lookup/268908*'
_output_shapes
:€€€€€€€€€ *
dtype02
embedding_3/embedding_lookupл
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_3/embedding_lookup/268908*'
_output_shapes
:€€€€€€€€€ 2'
%embedding_3/embedding_lookup/Identityј
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'embedding_3/embedding_lookup/Identity_1q
embedding_2/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€ДR2
embedding_2/CastМ
embedding_2/embedding_lookupResourceGather#embedding_2_embedding_lookup_268914embedding_2/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_2/embedding_lookup/268914*,
_output_shapes
:€€€€€€€€€ДR *
dtype02
embedding_2/embedding_lookupр
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_2/embedding_lookup/268914*,
_output_shapes
:€€€€€€€€€ДR 2'
%embedding_2/embedding_lookup/Identity≈
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2)
'embedding_2/embedding_lookup/Identity_1Ѓ
addAddV20embedding_2/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€ДR 2
addЮ
IdentityIdentityadd:z:0^embedding_2/embedding_lookup^embedding_3/embedding_lookup*
T0*,
_output_shapes
:€€€€€€€€€ДR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€ДR::2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup:K G
(
_output_shapes
:€€€€€€€€€ДR

_user_specified_namex
Ъ
ч
D__inference_conv1d_3_layer_call_and_return_conditional_losses_309502

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё 2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё *
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё *
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ё 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€ё 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€ё ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ё 
 
_user_specified_nameinputs
Љ0
»
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311417

inputs
assignmovingavg_311392
assignmovingavg_1_311398)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/311392*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_311392*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/311392*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/311392*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_311392AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/311392*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/311398*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_311398*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/311398*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/311398*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_311398AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/311398*
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
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/add_1Ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
Й•
щ
"__inference__traced_restore_312093
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
#assignvariableop_12_dense_11_kernel%
!assignvariableop_13_dense_11_bias'
#assignvariableop_14_dense_12_kernel%
!assignvariableop_15_dense_12_bias'
#assignvariableop_16_dense_13_kernel%
!assignvariableop_17_dense_13_biasM
Iassignvariableop_18_token_and_position_embedding_1_embedding_2_embeddingsM
Iassignvariableop_19_token_and_position_embedding_1_embedding_3_embeddingsO
Kassignvariableop_20_transformer_block_3_multi_head_attention_3_query_kernelM
Iassignvariableop_21_transformer_block_3_multi_head_attention_3_query_biasM
Iassignvariableop_22_transformer_block_3_multi_head_attention_3_key_kernelK
Gassignvariableop_23_transformer_block_3_multi_head_attention_3_key_biasO
Kassignvariableop_24_transformer_block_3_multi_head_attention_3_value_kernelM
Iassignvariableop_25_transformer_block_3_multi_head_attention_3_value_biasZ
Vassignvariableop_26_transformer_block_3_multi_head_attention_3_attention_output_kernelX
Tassignvariableop_27_transformer_block_3_multi_head_attention_3_attention_output_bias&
"assignvariableop_28_dense_9_kernel$
 assignvariableop_29_dense_9_bias'
#assignvariableop_30_dense_10_kernel%
!assignvariableop_31_dense_10_biasG
Cassignvariableop_32_transformer_block_3_layer_normalization_6_gammaF
Bassignvariableop_33_transformer_block_3_layer_normalization_6_betaG
Cassignvariableop_34_transformer_block_3_layer_normalization_7_gammaF
Bassignvariableop_35_transformer_block_3_layer_normalization_7_beta
assignvariableop_36_total
assignvariableop_37_count
identity_39ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9£
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*ѓ
value•BҐ'B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names№
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesс
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*≤
_output_shapesЯ
Ь:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_conv1d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1•
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2І
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4≥
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_2_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5≤
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_2_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6є
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_2_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7љ
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_2_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8≥
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_3_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9≤
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_3_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10љ
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_3_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ѕ
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_3_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ђ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_11_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_11_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ђ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_12_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_12_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ђ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_13_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_13_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18—
AssignVariableOp_18AssignVariableOpIassignvariableop_18_token_and_position_embedding_1_embedding_2_embeddingsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19—
AssignVariableOp_19AssignVariableOpIassignvariableop_19_token_and_position_embedding_1_embedding_3_embeddingsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20”
AssignVariableOp_20AssignVariableOpKassignvariableop_20_transformer_block_3_multi_head_attention_3_query_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21—
AssignVariableOp_21AssignVariableOpIassignvariableop_21_transformer_block_3_multi_head_attention_3_query_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22—
AssignVariableOp_22AssignVariableOpIassignvariableop_22_transformer_block_3_multi_head_attention_3_key_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ѕ
AssignVariableOp_23AssignVariableOpGassignvariableop_23_transformer_block_3_multi_head_attention_3_key_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24”
AssignVariableOp_24AssignVariableOpKassignvariableop_24_transformer_block_3_multi_head_attention_3_value_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25—
AssignVariableOp_25AssignVariableOpIassignvariableop_25_transformer_block_3_multi_head_attention_3_value_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ё
AssignVariableOp_26AssignVariableOpVassignvariableop_26_transformer_block_3_multi_head_attention_3_attention_output_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27№
AssignVariableOp_27AssignVariableOpTassignvariableop_27_transformer_block_3_multi_head_attention_3_attention_output_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28™
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_9_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29®
AssignVariableOp_29AssignVariableOp assignvariableop_29_dense_9_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ђ
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_10_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31©
AssignVariableOp_31AssignVariableOp!assignvariableop_31_dense_10_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ћ
AssignVariableOp_32AssignVariableOpCassignvariableop_32_transformer_block_3_layer_normalization_6_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33 
AssignVariableOp_33AssignVariableOpBassignvariableop_33_transformer_block_3_layer_normalization_6_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ћ
AssignVariableOp_34AssignVariableOpCassignvariableop_34_transformer_block_3_layer_normalization_7_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35 
AssignVariableOp_35AssignVariableOpBassignvariableop_35_transformer_block_3_layer_normalization_7_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36°
AssignVariableOp_36AssignVariableOpassignvariableop_36_totalIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37°
AssignVariableOp_37AssignVariableOpassignvariableop_37_countIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_379
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpҐ
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_38Х
Identity_39IdentityIdentity_38:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_39"#
identity_39Identity_39:output:0*ѓ
_input_shapesЭ
Ъ: ::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_37AssignVariableOp_372(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Љ0
»
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_309646

inputs
assignmovingavg_309621
assignmovingavg_1_309627)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/309621*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_309621*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/309621*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/309621*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_309621AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/309621*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/309627*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_309627*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/309627*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/309627*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_309627AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/309627*
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
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
batchnorm/add_1Ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
Ц
И
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_309292

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1и
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
…
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_311583

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
Ъ
ч
D__inference_conv1d_3_layer_call_and_return_conditional_losses_311126

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё 2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€ё *
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё *
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€ё 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€ё 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€ё 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€ё ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€ё 
 
_user_specified_nameinputs
у0
»
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311335

inputs
assignmovingavg_311310
assignmovingavg_1_311316)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/311310*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_311310*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/311310*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/311310*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_311310AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/311310*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/311316*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_311316*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/311316*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/311316*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_311316AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/311316*
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
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Й
П
?__inference_token_and_position_embedding_1_layer_call_fn_303104
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
 *,
_output_shapes
:€€€€€€€€€ДR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *c
f^R\
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_3030972
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€ДR 2

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
 
©
6__inference_batch_normalization_3_layer_call_fn_311463

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3096662
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
—
г
D__inference_dense_10_layer_call_and_return_conditional_losses_311822

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2	
BiasAddЬ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€#@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€#@
 
_user_specified_nameinputs
Иэ
÷
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_302859

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
identityИҐ.layer_normalization_6/batchnorm/ReadVariableOpҐ2layer_normalization_6/batchnorm/mul/ReadVariableOpҐ.layer_normalization_7/batchnorm/ReadVariableOpҐ2layer_normalization_7/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_3/attention_output/add/ReadVariableOpҐDmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_3/key/add/ReadVariableOpҐ7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_3/query/add/ReadVariableOpҐ9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_3/value/add/ReadVariableOpҐ9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpҐ,sequential_3/dense_10/BiasAdd/ReadVariableOpҐ.sequential_3/dense_10/Tensordot/ReadVariableOpҐ+sequential_3/dense_9/BiasAdd/ReadVariableOpҐ-sequential_3/dense_9/Tensordot/ReadVariableOpэ
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_3/query/einsum/EinsumEinsuminputsAmulti_head_attention_3/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2,
*multi_head_attention_3/query/einsum/Einsumџ
/multi_head_attention_3/query/add/ReadVariableOpReadVariableOp8multi_head_attention_3_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/query/add/ReadVariableOpх
 multi_head_attention_3/query/addAddV23multi_head_attention_3/query/einsum/Einsum:output:07multi_head_attention_3/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2"
 multi_head_attention_3/query/addч
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_3_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_3/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_3/key/einsum/EinsumEinsuminputs?multi_head_attention_3/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2*
(multi_head_attention_3/key/einsum/Einsum’
-multi_head_attention_3/key/add/ReadVariableOpReadVariableOp6multi_head_attention_3_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_3/key/add/ReadVariableOpн
multi_head_attention_3/key/addAddV21multi_head_attention_3/key/einsum/Einsum:output:05multi_head_attention_3/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2 
multi_head_attention_3/key/addэ
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_3_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_3/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_3/value/einsum/EinsumEinsuminputsAmulti_head_attention_3/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationabc,cde->abde2,
*multi_head_attention_3/value/einsum/Einsumџ
/multi_head_attention_3/value/add/ReadVariableOpReadVariableOp8multi_head_attention_3_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_3/value/add/ReadVariableOpх
 multi_head_attention_3/value/addAddV23multi_head_attention_3/value/einsum/Einsum:output:07multi_head_attention_3/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€# 2"
 multi_head_attention_3/value/addБ
multi_head_attention_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_3/Mul/y∆
multi_head_attention_3/MulMul$multi_head_attention_3/query/add:z:0%multi_head_attention_3/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€# 2
multi_head_attention_3/Mulь
$multi_head_attention_3/einsum/EinsumEinsum"multi_head_attention_3/key/add:z:0multi_head_attention_3/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€##*
equationaecd,abcd->acbe2&
$multi_head_attention_3/einsum/Einsumƒ
&multi_head_attention_3/softmax/SoftmaxSoftmax-multi_head_attention_3/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€##2(
&multi_head_attention_3/softmax/Softmax°
,multi_head_attention_3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,multi_head_attention_3/dropout/dropout/ConstВ
*multi_head_attention_3/dropout/dropout/MulMul0multi_head_attention_3/softmax/Softmax:softmax:05multi_head_attention_3/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€##2,
*multi_head_attention_3/dropout/dropout/MulЉ
,multi_head_attention_3/dropout/dropout/ShapeShape0multi_head_attention_3/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_3/dropout/dropout/ShapeЩ
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_3/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€##*
dtype02E
Cmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_3/dropout/dropout/GreaterEqual/y¬
3multi_head_attention_3/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_3/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_3/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€##25
3multi_head_attention_3/dropout/dropout/GreaterEqualд
+multi_head_attention_3/dropout/dropout/CastCast7multi_head_attention_3/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€##2-
+multi_head_attention_3/dropout/dropout/Castю
,multi_head_attention_3/dropout/dropout/Mul_1Mul.multi_head_attention_3/dropout/dropout/Mul:z:0/multi_head_attention_3/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€##2.
,multi_head_attention_3/dropout/dropout/Mul_1Ф
&multi_head_attention_3/einsum_1/EinsumEinsum0multi_head_attention_3/dropout/dropout/Mul_1:z:0$multi_head_attention_3/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€# *
equationacbe,aecd->abcd2(
&multi_head_attention_3/einsum_1/EinsumЮ
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_3_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_3/attention_output/einsum/EinsumEinsum/multi_head_attention_3/einsum_1/Einsum:output:0Lmulti_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€# *
equationabcd,cde->abe27
5multi_head_attention_3/attention_output/einsum/Einsumш
:multi_head_attention_3/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_3_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_3/attention_output/add/ReadVariableOpЭ
+multi_head_attention_3/attention_output/addAddV2>multi_head_attention_3/attention_output/einsum/Einsum:output:0Bmulti_head_attention_3/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2-
+multi_head_attention_3/attention_output/addw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_8/dropout/ConstЊ
dropout_8/dropout/MulMul/multi_head_attention_3/attention_output/add:z:0 dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_8/dropout/MulС
dropout_8/dropout/ShapeShape/multi_head_attention_3/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape÷
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€# *
dtype020
.dropout_8/dropout/random_uniform/RandomUniformЙ
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_8/dropout/GreaterEqual/yк
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2 
dropout_8/dropout/GreaterEqual°
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€# 2
dropout_8/dropout/Cast¶
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_8/dropout/Mul_1n
addAddV2inputsdropout_8/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
addґ
4layer_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_6/moments/mean/reduction_indicesя
"layer_normalization_6/moments/meanMeanadd:z:0=layer_normalization_6/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2$
"layer_normalization_6/moments/meanЋ
*layer_normalization_6/moments/StopGradientStopGradient+layer_normalization_6/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2,
*layer_normalization_6/moments/StopGradientл
/layer_normalization_6/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 21
/layer_normalization_6/moments/SquaredDifferenceЊ
8layer_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_6/moments/variance/reduction_indicesЧ
&layer_normalization_6/moments/varianceMean3layer_normalization_6/moments/SquaredDifference:z:0Alayer_normalization_6/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2(
&layer_normalization_6/moments/varianceУ
%layer_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_6/batchnorm/add/yк
#layer_normalization_6/batchnorm/addAddV2/layer_normalization_6/moments/variance:output:0.layer_normalization_6/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2%
#layer_normalization_6/batchnorm/addґ
%layer_normalization_6/batchnorm/RsqrtRsqrt'layer_normalization_6/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€#2'
%layer_normalization_6/batchnorm/Rsqrtа
2layer_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_6/batchnorm/mul/ReadVariableOpо
#layer_normalization_6/batchnorm/mulMul)layer_normalization_6/batchnorm/Rsqrt:y:0:layer_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_6/batchnorm/mulљ
%layer_normalization_6/batchnorm/mul_1Muladd:z:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/mul_1б
%layer_normalization_6/batchnorm/mul_2Mul+layer_normalization_6/moments/mean:output:0'layer_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/mul_2‘
.layer_normalization_6/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_6/batchnorm/ReadVariableOpк
#layer_normalization_6/batchnorm/subSub6layer_normalization_6/batchnorm/ReadVariableOp:value:0)layer_normalization_6/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_6/batchnorm/subб
%layer_normalization_6/batchnorm/add_1AddV2)layer_normalization_6/batchnorm/mul_1:z:0'layer_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_6/batchnorm/add_1’
-sequential_3/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_3_dense_9_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_3/dense_9/Tensordot/ReadVariableOpФ
#sequential_3/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_3/dense_9/Tensordot/axesЫ
#sequential_3/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_3/dense_9/Tensordot/free•
$sequential_3/dense_9/Tensordot/ShapeShape)layer_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/ShapeЮ
,sequential_3/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/GatherV2/axisЇ
'sequential_3/dense_9/Tensordot/GatherV2GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/free:output:05sequential_3/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/GatherV2Ґ
.sequential_3/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_3/dense_9/Tensordot/GatherV2_1/axisј
)sequential_3/dense_9/Tensordot/GatherV2_1GatherV2-sequential_3/dense_9/Tensordot/Shape:output:0,sequential_3/dense_9/Tensordot/axes:output:07sequential_3/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_3/dense_9/Tensordot/GatherV2_1Ц
$sequential_3/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_3/dense_9/Tensordot/Const‘
#sequential_3/dense_9/Tensordot/ProdProd0sequential_3/dense_9/Tensordot/GatherV2:output:0-sequential_3/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_3/dense_9/Tensordot/ProdЪ
&sequential_3/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_3/dense_9/Tensordot/Const_1№
%sequential_3/dense_9/Tensordot/Prod_1Prod2sequential_3/dense_9/Tensordot/GatherV2_1:output:0/sequential_3/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_3/dense_9/Tensordot/Prod_1Ъ
*sequential_3/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_3/dense_9/Tensordot/concat/axisЩ
%sequential_3/dense_9/Tensordot/concatConcatV2,sequential_3/dense_9/Tensordot/free:output:0,sequential_3/dense_9/Tensordot/axes:output:03sequential_3/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_9/Tensordot/concatа
$sequential_3/dense_9/Tensordot/stackPack,sequential_3/dense_9/Tensordot/Prod:output:0.sequential_3/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_3/dense_9/Tensordot/stackт
(sequential_3/dense_9/Tensordot/transpose	Transpose)layer_normalization_6/batchnorm/add_1:z:0.sequential_3/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2*
(sequential_3/dense_9/Tensordot/transposeу
&sequential_3/dense_9/Tensordot/ReshapeReshape,sequential_3/dense_9/Tensordot/transpose:y:0-sequential_3/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_3/dense_9/Tensordot/Reshapeт
%sequential_3/dense_9/Tensordot/MatMulMatMul/sequential_3/dense_9/Tensordot/Reshape:output:05sequential_3/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_3/dense_9/Tensordot/MatMulЪ
&sequential_3/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_3/dense_9/Tensordot/Const_2Ю
,sequential_3/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_3/dense_9/Tensordot/concat_1/axis¶
'sequential_3/dense_9/Tensordot/concat_1ConcatV20sequential_3/dense_9/Tensordot/GatherV2:output:0/sequential_3/dense_9/Tensordot/Const_2:output:05sequential_3/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_3/dense_9/Tensordot/concat_1д
sequential_3/dense_9/TensordotReshape/sequential_3/dense_9/Tensordot/MatMul:product:00sequential_3/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2 
sequential_3/dense_9/TensordotЋ
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_3/dense_9/BiasAdd/ReadVariableOpџ
sequential_3/dense_9/BiasAddBiasAdd'sequential_3/dense_9/Tensordot:output:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
sequential_3/dense_9/BiasAddЫ
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
sequential_3/dense_9/ReluЎ
.sequential_3/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_3_dense_10_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_3/dense_10/Tensordot/ReadVariableOpЦ
$sequential_3/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_3/dense_10/Tensordot/axesЭ
$sequential_3/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_3/dense_10/Tensordot/free•
%sequential_3/dense_10/Tensordot/ShapeShape'sequential_3/dense_9/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/Shape†
-sequential_3/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/GatherV2/axisњ
(sequential_3/dense_10/Tensordot/GatherV2GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/free:output:06sequential_3/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/GatherV2§
/sequential_3/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_3/dense_10/Tensordot/GatherV2_1/axis≈
*sequential_3/dense_10/Tensordot/GatherV2_1GatherV2.sequential_3/dense_10/Tensordot/Shape:output:0-sequential_3/dense_10/Tensordot/axes:output:08sequential_3/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_3/dense_10/Tensordot/GatherV2_1Ш
%sequential_3/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_3/dense_10/Tensordot/ConstЎ
$sequential_3/dense_10/Tensordot/ProdProd1sequential_3/dense_10/Tensordot/GatherV2:output:0.sequential_3/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_3/dense_10/Tensordot/ProdЬ
'sequential_3/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_1а
&sequential_3/dense_10/Tensordot/Prod_1Prod3sequential_3/dense_10/Tensordot/GatherV2_1:output:00sequential_3/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_3/dense_10/Tensordot/Prod_1Ь
+sequential_3/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_3/dense_10/Tensordot/concat/axisЮ
&sequential_3/dense_10/Tensordot/concatConcatV2-sequential_3/dense_10/Tensordot/free:output:0-sequential_3/dense_10/Tensordot/axes:output:04sequential_3/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_3/dense_10/Tensordot/concatд
%sequential_3/dense_10/Tensordot/stackPack-sequential_3/dense_10/Tensordot/Prod:output:0/sequential_3/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_3/dense_10/Tensordot/stackу
)sequential_3/dense_10/Tensordot/transpose	Transpose'sequential_3/dense_9/Relu:activations:0/sequential_3/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2+
)sequential_3/dense_10/Tensordot/transposeч
'sequential_3/dense_10/Tensordot/ReshapeReshape-sequential_3/dense_10/Tensordot/transpose:y:0.sequential_3/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_3/dense_10/Tensordot/Reshapeц
&sequential_3/dense_10/Tensordot/MatMulMatMul0sequential_3/dense_10/Tensordot/Reshape:output:06sequential_3/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_3/dense_10/Tensordot/MatMulЬ
'sequential_3/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_3/dense_10/Tensordot/Const_2†
-sequential_3/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_3/dense_10/Tensordot/concat_1/axisЂ
(sequential_3/dense_10/Tensordot/concat_1ConcatV21sequential_3/dense_10/Tensordot/GatherV2:output:00sequential_3/dense_10/Tensordot/Const_2:output:06sequential_3/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_3/dense_10/Tensordot/concat_1и
sequential_3/dense_10/TensordotReshape0sequential_3/dense_10/Tensordot/MatMul:product:01sequential_3/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2!
sequential_3/dense_10/Tensordotќ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_10/BiasAdd/ReadVariableOpя
sequential_3/dense_10/BiasAddBiasAdd(sequential_3/dense_10/Tensordot:output:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
sequential_3/dense_10/BiasAddw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_9/dropout/Constµ
dropout_9/dropout/MulMul&sequential_3/dense_10/BiasAdd:output:0 dropout_9/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_9/dropout/MulИ
dropout_9/dropout/ShapeShape&sequential_3/dense_10/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape÷
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€# *
dtype020
.dropout_9/dropout/random_uniform/RandomUniformЙ
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_9/dropout/GreaterEqual/yк
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2 
dropout_9/dropout/GreaterEqual°
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€# 2
dropout_9/dropout/Cast¶
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dropout_9/dropout/Mul_1Х
add_1AddV2)layer_normalization_6/batchnorm/add_1:z:0dropout_9/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
add_1ґ
4layer_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_7/moments/mean/reduction_indicesб
"layer_normalization_7/moments/meanMean	add_1:z:0=layer_normalization_7/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2$
"layer_normalization_7/moments/meanЋ
*layer_normalization_7/moments/StopGradientStopGradient+layer_normalization_7/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2,
*layer_normalization_7/moments/StopGradientн
/layer_normalization_7/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 21
/layer_normalization_7/moments/SquaredDifferenceЊ
8layer_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_7/moments/variance/reduction_indicesЧ
&layer_normalization_7/moments/varianceMean3layer_normalization_7/moments/SquaredDifference:z:0Alayer_normalization_7/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€#*
	keep_dims(2(
&layer_normalization_7/moments/varianceУ
%layer_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_7/batchnorm/add/yк
#layer_normalization_7/batchnorm/addAddV2/layer_normalization_7/moments/variance:output:0.layer_normalization_7/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€#2%
#layer_normalization_7/batchnorm/addґ
%layer_normalization_7/batchnorm/RsqrtRsqrt'layer_normalization_7/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€#2'
%layer_normalization_7/batchnorm/Rsqrtа
2layer_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_7/batchnorm/mul/ReadVariableOpо
#layer_normalization_7/batchnorm/mulMul)layer_normalization_7/batchnorm/Rsqrt:y:0:layer_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_7/batchnorm/mulњ
%layer_normalization_7/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/mul_1б
%layer_normalization_7/batchnorm/mul_2Mul+layer_normalization_7/moments/mean:output:0'layer_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/mul_2‘
.layer_normalization_7/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_7/batchnorm/ReadVariableOpк
#layer_normalization_7/batchnorm/subSub6layer_normalization_7/batchnorm/ReadVariableOp:value:0)layer_normalization_7/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2%
#layer_normalization_7/batchnorm/subб
%layer_normalization_7/batchnorm/add_1AddV2)layer_normalization_7/batchnorm/mul_1:z:0'layer_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2'
%layer_normalization_7/batchnorm/add_1’
IdentityIdentity)layer_normalization_7/batchnorm/add_1:z:0/^layer_normalization_6/batchnorm/ReadVariableOp3^layer_normalization_6/batchnorm/mul/ReadVariableOp/^layer_normalization_7/batchnorm/ReadVariableOp3^layer_normalization_7/batchnorm/mul/ReadVariableOp;^multi_head_attention_3/attention_output/add/ReadVariableOpE^multi_head_attention_3/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_3/key/add/ReadVariableOp8^multi_head_attention_3/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/query/add/ReadVariableOp:^multi_head_attention_3/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_3/value/add/ReadVariableOp:^multi_head_attention_3/value/einsum/Einsum/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp/^sequential_3/dense_10/Tensordot/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp.^sequential_3/dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€# ::::::::::::::::2`
.layer_normalization_6/batchnorm/ReadVariableOp.layer_normalization_6/batchnorm/ReadVariableOp2h
2layer_normalization_6/batchnorm/mul/ReadVariableOp2layer_normalization_6/batchnorm/mul/ReadVariableOp2`
.layer_normalization_7/batchnorm/ReadVariableOp.layer_normalization_7/batchnorm/ReadVariableOp2h
2layer_normalization_7/batchnorm/mul/ReadVariableOp2layer_normalization_7/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_3/attention_output/add/ReadVariableOp:multi_head_attention_3/attention_output/add/ReadVariableOp2М
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
:€€€€€€€€€# 
 
_user_specified_nameinputs
–

а
4__inference_transformer_block_3_layer_call_fn_301561

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
 *+
_output_shapes
:€€€€€€€€€# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_3015402
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
µ
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_311481

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€а2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€а2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€# :S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
±I
Ђ
H__inference_sequential_3_layer_call_and_return_conditional_losses_311669

inputs-
)dense_9_tensordot_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource.
*dense_10_tensordot_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource
identityИҐdense_10/BiasAdd/ReadVariableOpҐ!dense_10/Tensordot/ReadVariableOpҐdense_9/BiasAdd/ReadVariableOpҐ dense_9/Tensordot/ReadVariableOpЃ
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
dense_9/Tensordot/axesБ
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
dense_9/Tensordot/ShapeД
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisщ
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2И
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axis€
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
dense_9/Tensordot/Const†
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/ProdА
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1®
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1А
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisЎ
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatђ
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stack®
dense_9/Tensordot/transpose	Transposeinputs!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dense_9/Tensordot/transposeњ
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_9/Tensordot/ReshapeЊ
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_9/Tensordot/MatMulА
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_9/Tensordot/Const_2Д
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisе
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1∞
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
dense_9/Tensordot§
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_9/BiasAdd/ReadVariableOpІ
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
dense_9/BiasAddt
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
dense_9/Relu±
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
dense_10/Tensordot/axesГ
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
dense_10/Tensordot/ShapeЖ
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axisю
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2К
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axisД
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
dense_10/Tensordot/Const§
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/ProdВ
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1ђ
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1В
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axisЁ
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat∞
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stackњ
dense_10/Tensordot/transpose	Transposedense_9/Relu:activations:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
dense_10/Tensordot/transpose√
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_10/Tensordot/Reshape¬
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_10/Tensordot/MatMulВ
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_2Ж
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisк
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1і
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dense_10/TensordotІ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_10/BiasAdd/ReadVariableOpЂ
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
dense_10/BiasAddы
IdentityIdentitydense_10/BiasAdd:output:0 ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
∞ 
в
C__inference_dense_9_layer_call_and_return_conditional_losses_311783

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€# 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€#@2
ReluЮ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€# ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
Д
P
4__inference_average_pooling1d_4_layer_call_fn_309148

inputs
identityж
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
GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_3091422
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
Ц
И
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311355

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1и
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
°
F
*__inference_flatten_1_layer_call_fn_311486

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€а* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_3097562
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€а2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€# :S O
+
_output_shapes
:€€€€€€€€€# 
 
_user_specified_nameinputs
њ
m
A__inference_add_1_layer_call_and_return_conditional_losses_311469
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:€€€€€€€€€# 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:€€€€€€€€€# :€€€€€€€€€# :U Q
+
_output_shapes
:€€€€€€€€€# 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€# 
"
_user_specified_name
inputs/1
Ќ
І
-__inference_sequential_3_layer_call_fn_310580
dense_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCalldense_9_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_3105692
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€# ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€# 
'
_user_specified_namedense_9_input"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*й
serving_default’
<
input_31
serving_default_input_3:0€€€€€€€€€ДR
;
input_40
serving_default_input_4:0€€€€€€€€€<
dense_130
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:÷ы
тG
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

signatures
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
±__call__
≤_default_save_signature
+≥&call_and_return_all_conditional_losses"ХB
_tf_keras_networkщA{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_1", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["token_and_position_embedding_1", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_3", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["average_pooling1d_3", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_4", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_5", "inbound_nodes": [[["token_and_position_embedding_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["average_pooling1d_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["average_pooling1d_5", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}], ["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_3", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["transformer_block_3", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["dense_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dropout_10", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_11", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dropout_11", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["dense_13", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.00020000000949949026, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
Ц
#_self_saveable_object_factories"о
_tf_keras_input_layerќ{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
М
	token_emb
pos_emb
#_self_saveable_object_factories
trainable_variables
 regularization_losses
!	variables
"	keras_api
і__call__
+µ&call_and_return_all_conditional_losses"Ї
_tf_keras_layer†{"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
О


#kernel
$bias
#%_self_saveable_object_factories
&trainable_variables
'regularization_losses
(	variables
)	keras_api
ґ__call__
+Ј&call_and_return_all_conditional_losses"¬
_tf_keras_layer®{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500, 32]}}
Ѓ
#*_self_saveable_object_factories
+trainable_variables
,regularization_losses
-	variables
.	keras_api
Є__call__
+є&call_and_return_all_conditional_losses"ш
_tf_keras_layerё{"class_name": "AveragePooling1D", "name": "average_pooling1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
М


/kernel
0bias
#1_self_saveable_object_factories
2trainable_variables
3regularization_losses
4	variables
5	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses"ј
_tf_keras_layer¶{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 350, 32]}}
Ѓ
#6_self_saveable_object_factories
7trainable_variables
8regularization_losses
9	variables
:	keras_api
Љ__call__
+љ&call_and_return_all_conditional_losses"ш
_tf_keras_layerё{"class_name": "AveragePooling1D", "name": "average_pooling1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_4", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
∞
#;_self_saveable_object_factories
<trainable_variables
=regularization_losses
>	variables
?	keras_api
Њ__call__
+њ&call_and_return_all_conditional_losses"ъ
_tf_keras_layerа{"class_name": "AveragePooling1D", "name": "average_pooling1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_5", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ё	
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
#E_self_saveable_object_factories
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses"в
_tf_keras_layer»{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Ё	
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
#O_self_saveable_object_factories
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
¬__call__
+√&call_and_return_all_conditional_losses"в
_tf_keras_layer»{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Ў
#T_self_saveable_object_factories
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
ƒ__call__
+≈&call_and_return_all_conditional_losses"Ґ
_tf_keras_layerИ{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 35, 32]}, {"class_name": "TensorShape", "items": [null, 35, 32]}]}
©
Yatt
Zffn
[
layernorm1
\
layernorm2
]dropout1
^dropout2
#__self_saveable_object_factories
`trainable_variables
aregularization_losses
b	variables
c	keras_api
∆__call__
+«&call_and_return_all_conditional_losses"•
_tf_keras_layerЛ{"class_name": "TransformerBlock", "name": "transformer_block_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Н
#d_self_saveable_object_factories
etrainable_variables
fregularization_losses
g	variables
h	keras_api
»__call__
+…&call_and_return_all_conditional_losses"„
_tf_keras_layerљ{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
О
#i_self_saveable_object_factories"ж
_tf_keras_input_layer∆{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
х
#j_self_saveable_object_factories
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
 __call__
+Ћ&call_and_return_all_conditional_losses"њ
_tf_keras_layer•{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1120]}, {"class_name": "TensorShape", "items": [null, 8]}]}
Э

okernel
pbias
#q_self_saveable_object_factories
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
ћ__call__
+Ќ&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1128]}}
О
#v_self_saveable_object_factories
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
Ы

{kernel
|bias
#}_self_saveable_object_factories
~trainable_variables
regularization_losses
А	variables
Б	keras_api
–__call__
+—&call_and_return_all_conditional_losses"Ќ
_tf_keras_layer≥{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
У
$В_self_saveable_object_factories
Гtrainable_variables
Дregularization_losses
Е	variables
Ж	keras_api
“__call__
+”&call_and_return_all_conditional_losses"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
°
Зkernel
	Иbias
$Й_self_saveable_object_factories
Кtrainable_variables
Лregularization_losses
М	variables
Н	keras_api
‘__call__
+’&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
"
	optimizer
-
÷serving_default"
signature_map
 "
trackable_dict_wrapper
™
О0
П1
#2
$3
/4
05
A6
B7
K8
L9
Р10
С11
Т12
У13
Ф14
Х15
Ц16
Ч17
Ш18
Щ19
Ъ20
Ы21
Ь22
Э23
Ю24
Я25
o26
p27
{28
|29
З30
И31"
trackable_list_wrapper
 "
trackable_list_wrapper
 
О0
П1
#2
$3
/4
05
A6
B7
C8
D9
K10
L11
M12
N13
Р14
С15
Т16
У17
Ф18
Х19
Ц20
Ч21
Ш22
Щ23
Ъ24
Ы25
Ь26
Э27
Ю28
Я29
o30
p31
{32
|33
З34
И35"
trackable_list_wrapper
”
†layer_metrics
trainable_variables
 °layer_regularization_losses
regularization_losses
	variables
Ґlayers
£non_trainable_variables
§metrics
±__call__
≤_default_save_signature
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
џ
О
embeddings
$•_self_saveable_object_factories
¶trainable_variables
Іregularization_losses
®	variables
©	keras_api
„__call__
+Ў&call_and_return_all_conditional_losses"П
_tf_keras_layerх{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}}
Ў
П
embeddings
$™_self_saveable_object_factories
Ђtrainable_variables
ђregularization_losses
≠	variables
Ѓ	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses"М
_tf_keras_layerт{"class_name": "Embedding", "name": "embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
 "
trackable_dict_wrapper
0
О0
П1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
µ
ѓlayer_metrics
trainable_variables
 ∞layer_regularization_losses
 regularization_losses
!	variables
±layers
≤non_trainable_variables
≥metrics
і__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
%:#  2conv1d_2/kernel
: 2conv1d_2/bias
 "
trackable_dict_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
µ
іlayer_metrics
&trainable_variables
 µlayer_regularization_losses
'regularization_losses
(	variables
ґlayers
Јnon_trainable_variables
Єmetrics
ґ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
єlayer_metrics
+trainable_variables
 Їlayer_regularization_losses
,regularization_losses
-	variables
їlayers
Љnon_trainable_variables
љmetrics
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
%:#	  2conv1d_3/kernel
: 2conv1d_3/bias
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
µ
Њlayer_metrics
2trainable_variables
 њlayer_regularization_losses
3regularization_losses
4	variables
јlayers
Ѕnon_trainable_variables
¬metrics
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
√layer_metrics
7trainable_variables
 ƒlayer_regularization_losses
8regularization_losses
9	variables
≈layers
∆non_trainable_variables
«metrics
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
»layer_metrics
<trainable_variables
 …layer_regularization_losses
=regularization_losses
>	variables
 layers
Ћnon_trainable_variables
ћmetrics
Њ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_2/gamma
(:& 2batch_normalization_2/beta
1:/  (2!batch_normalization_2/moving_mean
5:3  (2%batch_normalization_2/moving_variance
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
A0
B1
C2
D3"
trackable_list_wrapper
µ
Ќlayer_metrics
Ftrainable_variables
 ќlayer_regularization_losses
Gregularization_losses
H	variables
ѕlayers
–non_trainable_variables
—metrics
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
 "
trackable_dict_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
K0
L1
M2
N3"
trackable_list_wrapper
µ
“layer_metrics
Ptrainable_variables
 ”layer_regularization_losses
Qregularization_losses
R	variables
‘layers
’non_trainable_variables
÷metrics
¬__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
„layer_metrics
Utrainable_variables
 Ўlayer_regularization_losses
Vregularization_losses
W	variables
ўlayers
Џnon_trainable_variables
џmetrics
ƒ__call__
+≈&call_and_return_all_conditional_losses
'≈"call_and_return_conditional_losses"
_generic_user_object
Ѓ
№_query_dense
Ё
_key_dense
ё_value_dense
я_softmax
а_dropout_layer
б_output_dense
$в_self_saveable_object_factories
гtrainable_variables
дregularization_losses
е	variables
ж	keras_api
џ__call__
+№&call_and_return_all_conditional_losses"Д
_tf_keras_layerк{"class_name": "MultiHeadAttention", "name": "multi_head_attention_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_3", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
—
зlayer_with_weights-0
зlayer-0
иlayer_with_weights-1
иlayer-1
$й_self_saveable_object_factories
кtrainable_variables
лregularization_losses
м	variables
н	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses"ƒ
_tf_keras_sequential•{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_9_input"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_9_input"}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
Р
	оaxis

Ьgamma
	Эbeta
$п_self_saveable_object_factories
рtrainable_variables
сregularization_losses
т	variables
у	keras_api
я__call__
+а&call_and_return_all_conditional_losses"≥
_tf_keras_layerЩ{"class_name": "LayerNormalization", "name": "layer_normalization_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Р
	фaxis

Юgamma
	Яbeta
$х_self_saveable_object_factories
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
б__call__
+в&call_and_return_all_conditional_losses"≥
_tf_keras_layerЩ{"class_name": "LayerNormalization", "name": "layer_normalization_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
С
$ъ_self_saveable_object_factories
ыtrainable_variables
ьregularization_losses
э	variables
ю	keras_api
г__call__
+д&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
С
$€_self_saveable_object_factories
Аtrainable_variables
Бregularization_losses
В	variables
Г	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
 "
trackable_dict_wrapper
¶
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
Ш8
Щ9
Ъ10
Ы11
Ь12
Э13
Ю14
Я15"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7
Ш8
Щ9
Ъ10
Ы11
Ь12
Э13
Ю14
Я15"
trackable_list_wrapper
µ
Дlayer_metrics
`trainable_variables
 Еlayer_regularization_losses
aregularization_losses
b	variables
Жlayers
Зnon_trainable_variables
Иmetrics
∆__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Йlayer_metrics
etrainable_variables
 Кlayer_regularization_losses
fregularization_losses
g	variables
Лlayers
Мnon_trainable_variables
Нmetrics
»__call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Оlayer_metrics
ktrainable_variables
 Пlayer_regularization_losses
lregularization_losses
m	variables
Рlayers
Сnon_trainable_variables
Тmetrics
 __call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
": 	и@2dense_11/kernel
:@2dense_11/bias
 "
trackable_dict_wrapper
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
µ
Уlayer_metrics
rtrainable_variables
 Фlayer_regularization_losses
sregularization_losses
t	variables
Хlayers
Цnon_trainable_variables
Чmetrics
ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Шlayer_metrics
wtrainable_variables
 Щlayer_regularization_losses
xregularization_losses
y	variables
Ъlayers
Ыnon_trainable_variables
Ьmetrics
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_12/kernel
:@2dense_12/bias
 "
trackable_dict_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
ґ
Эlayer_metrics
~trainable_variables
 Юlayer_regularization_losses
regularization_losses
А	variables
Яlayers
†non_trainable_variables
°metrics
–__call__
+—&call_and_return_all_conditional_losses
'—"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ґlayer_metrics
Гtrainable_variables
 £layer_regularization_losses
Дregularization_losses
Е	variables
§layers
•non_trainable_variables
¶metrics
“__call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_13/kernel
:2dense_13/bias
 "
trackable_dict_wrapper
0
З0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
З0
И1"
trackable_list_wrapper
Є
Іlayer_metrics
Кtrainable_variables
 ®layer_regularization_losses
Лregularization_losses
М	variables
©layers
™non_trainable_variables
Ђmetrics
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
G:E 25token_and_position_embedding_1/embedding_2/embeddings
H:F	ДR 25token_and_position_embedding_1/embedding_3/embeddings
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
trackable_dict_wrapper
 "
trackable_list_wrapper
Ѓ
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
<
C0
D1
M2
N3"
trackable_list_wrapper
(
ђ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
О0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
О0"
trackable_list_wrapper
Є
≠layer_metrics
¶trainable_variables
 Ѓlayer_regularization_losses
Іregularization_losses
®	variables
ѓlayers
∞non_trainable_variables
±metrics
„__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
(
П0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
П0"
trackable_list_wrapper
Є
≤layer_metrics
Ђtrainable_variables
 ≥layer_regularization_losses
ђregularization_losses
≠	variables
іlayers
µnon_trainable_variables
ґmetrics
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
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
.
C0
D1"
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
M0
N1"
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
с
Јpartial_output_shape
Єfull_output_shape
Рkernel
	Сbias
$є_self_saveable_object_factories
Їtrainable_variables
їregularization_losses
Љ	variables
љ	keras_api
з__call__
+и&call_and_return_all_conditional_losses"л
_tf_keras_layer—{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
н
Њpartial_output_shape
њfull_output_shape
Тkernel
	Уbias
$ј_self_saveable_object_factories
Ѕtrainable_variables
¬regularization_losses
√	variables
ƒ	keras_api
й__call__
+к&call_and_return_all_conditional_losses"з
_tf_keras_layerЌ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
с
≈partial_output_shape
∆full_output_shape
Фkernel
	Хbias
$«_self_saveable_object_factories
»trainable_variables
…regularization_losses
 	variables
Ћ	keras_api
л__call__
+м&call_and_return_all_conditional_losses"л
_tf_keras_layer—{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
С
$ћ_self_saveable_object_factories
Ќtrainable_variables
ќregularization_losses
ѕ	variables
–	keras_api
н__call__
+о&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
Н
$—_self_saveable_object_factories
“trainable_variables
”regularization_losses
‘	variables
’	keras_api
п__call__
+р&call_and_return_all_conditional_losses"“
_tf_keras_layerЄ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
Ж
÷partial_output_shape
„full_output_shape
Цkernel
	Чbias
$Ў_self_saveable_object_factories
ўtrainable_variables
Џregularization_losses
џ	variables
№	keras_api
с__call__
+т&call_and_return_all_conditional_losses"А
_tf_keras_layerж{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 1, 32]}}
 "
trackable_dict_wrapper
`
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
Р0
С1
Т2
У3
Ф4
Х5
Ц6
Ч7"
trackable_list_wrapper
Є
Ёlayer_metrics
гtrainable_variables
 ёlayer_regularization_losses
дregularization_losses
е	variables
яlayers
аnon_trainable_variables
бmetrics
џ__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
Ґ
Шkernel
	Щbias
$в_self_saveable_object_factories
гtrainable_variables
дregularization_losses
е	variables
ж	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
¶
Ъkernel
	Ыbias
$з_self_saveable_object_factories
иtrainable_variables
йregularization_losses
к	variables
л	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"”
_tf_keras_layerє{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 64]}}
 "
trackable_dict_wrapper
@
Ш0
Щ1
Ъ2
Ы3"
trackable_list_wrapper
 "
trackable_list_wrapper
@
Ш0
Щ1
Ъ2
Ы3"
trackable_list_wrapper
Є
мlayer_metrics
кtrainable_variables
 нlayer_regularization_losses
лregularization_losses
м	variables
оlayers
пnon_trainable_variables
рmetrics
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ь0
Э1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ь0
Э1"
trackable_list_wrapper
Є
сlayer_metrics
рtrainable_variables
 тlayer_regularization_losses
сregularization_losses
т	variables
уlayers
фnon_trainable_variables
хmetrics
я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ю0
Я1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ю0
Я1"
trackable_list_wrapper
Є
цlayer_metrics
цtrainable_variables
 чlayer_regularization_losses
чregularization_losses
ш	variables
шlayers
щnon_trainable_variables
ъmetrics
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ыlayer_metrics
ыtrainable_variables
 ьlayer_regularization_losses
ьregularization_losses
э	variables
эlayers
юnon_trainable_variables
€metrics
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Аlayer_metrics
Аtrainable_variables
 Бlayer_regularization_losses
Бregularization_losses
В	variables
Вlayers
Гnon_trainable_variables
Дmetrics
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
Y0
Z1
[2
\3
]4
^5"
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
њ

Еtotal

Жcount
З	variables
И	keras_api"Д
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
 "
trackable_dict_wrapper
0
Р0
С1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
Є
Йlayer_metrics
Їtrainable_variables
 Кlayer_regularization_losses
їregularization_losses
Љ	variables
Лlayers
Мnon_trainable_variables
Нmetrics
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Т0
У1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Т0
У1"
trackable_list_wrapper
Є
Оlayer_metrics
Ѕtrainable_variables
 Пlayer_regularization_losses
¬regularization_losses
√	variables
Рlayers
Сnon_trainable_variables
Тmetrics
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ф0
Х1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ф0
Х1"
trackable_list_wrapper
Є
Уlayer_metrics
»trainable_variables
 Фlayer_regularization_losses
…regularization_losses
 	variables
Хlayers
Цnon_trainable_variables
Чmetrics
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Шlayer_metrics
Ќtrainable_variables
 Щlayer_regularization_losses
ќregularization_losses
ѕ	variables
Ъlayers
Ыnon_trainable_variables
Ьmetrics
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Эlayer_metrics
“trainable_variables
 Юlayer_regularization_losses
”regularization_losses
‘	variables
Яlayers
†non_trainable_variables
°metrics
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ц0
Ч1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ц0
Ч1"
trackable_list_wrapper
Є
Ґlayer_metrics
ўtrainable_variables
 £layer_regularization_losses
Џregularization_losses
џ	variables
§layers
•non_trainable_variables
¶metrics
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
P
№0
Ё1
ё2
я3
а4
б5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ш0
Щ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ш0
Щ1"
trackable_list_wrapper
Є
Іlayer_metrics
гtrainable_variables
 ®layer_regularization_losses
дregularization_losses
е	variables
©layers
™non_trainable_variables
Ђmetrics
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
0
Ъ0
Ы1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ъ0
Ы1"
trackable_list_wrapper
Є
ђlayer_metrics
иtrainable_variables
 ≠layer_regularization_losses
йregularization_losses
к	variables
Ѓlayers
ѓnon_trainable_variables
∞metrics
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
з0
и1"
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
Е0
Ж1"
trackable_list_wrapper
.
З	variables"
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
о2л
(__inference_model_1_layer_call_fn_310188
(__inference_model_1_layer_call_fn_311085
(__inference_model_1_layer_call_fn_310360
(__inference_model_1_layer_call_fn_311007ј
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
!__inference__wrapped_model_309118я
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
annotations™ *OҐL
JЪG
"К
input_3€€€€€€€€€ДR
!К
input_4€€€€€€€€€
Џ2„
C__inference_model_1_layer_call_and_return_conditional_losses_310791
C__inference_model_1_layer_call_and_return_conditional_losses_310015
C__inference_model_1_layer_call_and_return_conditional_losses_309921
C__inference_model_1_layer_call_and_return_conditional_losses_310929ј
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
Џ2„
?__inference_token_and_position_embedding_1_layer_call_fn_303104У
М≤И
FullArgSpec
argsЪ
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
annotations™ *
 
х2т
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_301033У
М≤И
FullArgSpec
argsЪ
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
annotations™ *
 
”2–
)__inference_conv1d_2_layer_call_fn_311110Ґ
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
D__inference_conv1d_2_layer_call_and_return_conditional_losses_311101Ґ
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
П2М
4__inference_average_pooling1d_3_layer_call_fn_309133”
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
™2І
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_309127”
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
”2–
)__inference_conv1d_3_layer_call_fn_311135Ґ
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
D__inference_conv1d_3_layer_call_and_return_conditional_losses_311126Ґ
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
П2М
4__inference_average_pooling1d_4_layer_call_fn_309148”
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
™2І
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_309142”
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
П2М
4__inference_average_pooling1d_5_layer_call_fn_309163”
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
™2І
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_309157”
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
Ъ2Ч
6__inference_batch_normalization_2_layer_call_fn_311286
6__inference_batch_normalization_2_layer_call_fn_311299
6__inference_batch_normalization_2_layer_call_fn_311217
6__inference_batch_normalization_2_layer_call_fn_311204і
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
Ж2Г
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311191
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311253
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311171
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311273і
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
Ъ2Ч
6__inference_batch_normalization_3_layer_call_fn_311463
6__inference_batch_normalization_3_layer_call_fn_311381
6__inference_batch_normalization_3_layer_call_fn_311368
6__inference_batch_normalization_3_layer_call_fn_311450і
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
Ж2Г
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311355
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311417
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311335
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311437і
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
–2Ќ
&__inference_add_1_layer_call_fn_311475Ґ
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
л2и
A__inference_add_1_layer_call_and_return_conditional_losses_311469Ґ
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
Ш2Х
4__inference_transformer_block_3_layer_call_fn_303075
4__inference_transformer_block_3_layer_call_fn_301561¶
Я≤Ы
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ќ2Ћ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_302859
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_302342¶
Я≤Ы
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
‘2—
*__inference_flatten_1_layer_call_fn_311486Ґ
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
п2м
E__inference_flatten_1_layer_call_and_return_conditional_losses_311481Ґ
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
Ў2’
.__inference_concatenate_1_layer_call_fn_311499Ґ
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
у2р
I__inference_concatenate_1_layer_call_and_return_conditional_losses_311493Ґ
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
”2–
)__inference_dense_11_layer_call_fn_311519Ґ
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
D__inference_dense_11_layer_call_and_return_conditional_losses_311510Ґ
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
Ф2С
+__inference_dropout_10_layer_call_fn_311541
+__inference_dropout_10_layer_call_fn_311546і
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
 2«
F__inference_dropout_10_layer_call_and_return_conditional_losses_311536
F__inference_dropout_10_layer_call_and_return_conditional_losses_311531і
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
”2–
)__inference_dense_12_layer_call_fn_311566Ґ
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
D__inference_dense_12_layer_call_and_return_conditional_losses_311557Ґ
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
Ф2С
+__inference_dropout_11_layer_call_fn_311588
+__inference_dropout_11_layer_call_fn_311593і
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
 2«
F__inference_dropout_11_layer_call_and_return_conditional_losses_311578
F__inference_dropout_11_layer_call_and_return_conditional_losses_311583і
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
”2–
)__inference_dense_13_layer_call_fn_311612Ґ
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
D__inference_dense_13_layer_call_and_return_conditional_losses_311603Ґ
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
$__inference_signature_wrapper_310440input_3input_4"Ф
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
В2€
-__inference_sequential_3_layer_call_fn_311752
-__inference_sequential_3_layer_call_fn_311739
-__inference_sequential_3_layer_call_fn_310607
-__inference_sequential_3_layer_call_fn_310580ј
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
о2л
H__inference_sequential_3_layer_call_and_return_conditional_losses_310552
H__inference_sequential_3_layer_call_and_return_conditional_losses_311726
H__inference_sequential_3_layer_call_and_return_conditional_losses_311669
H__inference_sequential_3_layer_call_and_return_conditional_losses_310538ј
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
(__inference_dense_9_layer_call_fn_311792Ґ
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
C__inference_dense_9_layer_call_and_return_conditional_losses_311783Ґ
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
”2–
)__inference_dense_10_layer_call_fn_311831Ґ
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
D__inference_dense_10_layer_call_and_return_conditional_losses_311822Ґ
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
 р
!__inference__wrapped_model_309118 8ПО#$/0DACBNKMLРСТУФХЦЧЬЭШЩЪЫЮЯop{|ЗИYҐV
OҐL
JЪG
"К
input_3€€€€€€€€€ДR
!К
input_4€€€€€€€€€
™ "3™0
.
dense_13"К
dense_13€€€€€€€€€’
A__inference_add_1_layer_call_and_return_conditional_losses_311469ПbҐ_
XҐU
SЪP
&К#
inputs/0€€€€€€€€€# 
&К#
inputs/1€€€€€€€€€# 
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ ≠
&__inference_add_1_layer_call_fn_311475ВbҐ_
XҐU
SЪP
&К#
inputs/0€€€€€€€€€# 
&К#
inputs/1€€€€€€€€€# 
™ "К€€€€€€€€€# Ў
O__inference_average_pooling1d_3_layer_call_and_return_conditional_losses_309127ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѓ
4__inference_average_pooling1d_3_layer_call_fn_309133wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€Ў
O__inference_average_pooling1d_4_layer_call_and_return_conditional_losses_309142ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѓ
4__inference_average_pooling1d_4_layer_call_fn_309148wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€Ў
O__inference_average_pooling1d_5_layer_call_and_return_conditional_losses_309157ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѓ
4__inference_average_pooling1d_5_layer_call_fn_309163wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€њ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311171jCDAB7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ њ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311191jDACB7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p 
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ —
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311253|CDAB@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ —
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_311273|DACB@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ Ч
6__inference_batch_normalization_2_layer_call_fn_311204]CDAB7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p
™ "К€€€€€€€€€# Ч
6__inference_batch_normalization_2_layer_call_fn_311217]DACB7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p 
™ "К€€€€€€€€€# ©
6__inference_batch_normalization_2_layer_call_fn_311286oCDAB@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p
™ "%К"€€€€€€€€€€€€€€€€€€ ©
6__inference_batch_normalization_2_layer_call_fn_311299oDACB@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p 
™ "%К"€€€€€€€€€€€€€€€€€€ —
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311335|MNKL@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ —
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311355|NKML@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ њ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311417jMNKL7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ њ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_311437jNKML7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p 
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ ©
6__inference_batch_normalization_3_layer_call_fn_311368oMNKL@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p
™ "%К"€€€€€€€€€€€€€€€€€€ ©
6__inference_batch_normalization_3_layer_call_fn_311381oNKML@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p 
™ "%К"€€€€€€€€€€€€€€€€€€ Ч
6__inference_batch_normalization_3_layer_call_fn_311450]MNKL7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p
™ "К€€€€€€€€€# Ч
6__inference_batch_normalization_3_layer_call_fn_311463]NKML7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p 
™ "К€€€€€€€€€# ”
I__inference_concatenate_1_layer_call_and_return_conditional_losses_311493Е[ҐX
QҐN
LЪI
#К 
inputs/0€€€€€€€€€а
"К
inputs/1€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€и
Ъ ™
.__inference_concatenate_1_layer_call_fn_311499x[ҐX
QҐN
LЪI
#К 
inputs/0€€€€€€€€€а
"К
inputs/1€€€€€€€€€
™ "К€€€€€€€€€иЃ
D__inference_conv1d_2_layer_call_and_return_conditional_losses_311101f#$4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ДR 
™ "*Ґ'
 К
0€€€€€€€€€ДR 
Ъ Ж
)__inference_conv1d_2_layer_call_fn_311110Y#$4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ДR 
™ "К€€€€€€€€€ДR Ѓ
D__inference_conv1d_3_layer_call_and_return_conditional_losses_311126f/04Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ё 
™ "*Ґ'
 К
0€€€€€€€€€ё 
Ъ Ж
)__inference_conv1d_3_layer_call_fn_311135Y/04Ґ1
*Ґ'
%К"
inputs€€€€€€€€€ё 
™ "К€€€€€€€€€ё Ѓ
D__inference_dense_10_layer_call_and_return_conditional_losses_311822fЪЫ3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€#@
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ Ж
)__inference_dense_10_layer_call_fn_311831YЪЫ3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€#@
™ "К€€€€€€€€€# •
D__inference_dense_11_layer_call_and_return_conditional_losses_311510]op0Ґ-
&Ґ#
!К
inputs€€€€€€€€€и
™ "%Ґ"
К
0€€€€€€€€€@
Ъ }
)__inference_dense_11_layer_call_fn_311519Pop0Ґ-
&Ґ#
!К
inputs€€€€€€€€€и
™ "К€€€€€€€€€@§
D__inference_dense_12_layer_call_and_return_conditional_losses_311557\{|/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
)__inference_dense_12_layer_call_fn_311566O{|/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@¶
D__inference_dense_13_layer_call_and_return_conditional_losses_311603^ЗИ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€
Ъ ~
)__inference_dense_13_layer_call_fn_311612QЗИ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€≠
C__inference_dense_9_layer_call_and_return_conditional_losses_311783fШЩ3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€# 
™ ")Ґ&
К
0€€€€€€€€€#@
Ъ Е
(__inference_dense_9_layer_call_fn_311792YШЩ3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€# 
™ "К€€€€€€€€€#@¶
F__inference_dropout_10_layer_call_and_return_conditional_losses_311531\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ¶
F__inference_dropout_10_layer_call_and_return_conditional_losses_311536\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ~
+__inference_dropout_10_layer_call_fn_311541O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@~
+__inference_dropout_10_layer_call_fn_311546O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@¶
F__inference_dropout_11_layer_call_and_return_conditional_losses_311578\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ¶
F__inference_dropout_11_layer_call_and_return_conditional_losses_311583\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ~
+__inference_dropout_11_layer_call_fn_311588O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@~
+__inference_dropout_11_layer_call_fn_311593O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@¶
E__inference_flatten_1_layer_call_and_return_conditional_losses_311481]3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€# 
™ "&Ґ#
К
0€€€€€€€€€а
Ъ ~
*__inference_flatten_1_layer_call_fn_311486P3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€# 
™ "К€€€€€€€€€аМ
C__inference_model_1_layer_call_and_return_conditional_losses_309921ƒ8ПО#$/0CDABMNKLРСТУФХЦЧЬЭШЩЪЫЮЯop{|ЗИaҐ^
WҐT
JЪG
"К
input_3€€€€€€€€€ДR
!К
input_4€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ М
C__inference_model_1_layer_call_and_return_conditional_losses_310015ƒ8ПО#$/0DACBNKMLРСТУФХЦЧЬЭШЩЪЫЮЯop{|ЗИaҐ^
WҐT
JЪG
"К
input_3€€€€€€€€€ДR
!К
input_4€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ О
C__inference_model_1_layer_call_and_return_conditional_losses_310791∆8ПО#$/0CDABMNKLРСТУФХЦЧЬЭШЩЪЫЮЯop{|ЗИcҐ`
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
Ъ О
C__inference_model_1_layer_call_and_return_conditional_losses_310929∆8ПО#$/0DACBNKMLРСТУФХЦЧЬЭШЩЪЫЮЯop{|ЗИcҐ`
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
Ъ д
(__inference_model_1_layer_call_fn_310188Ј8ПО#$/0CDABMNKLРСТУФХЦЧЬЭШЩЪЫЮЯop{|ЗИaҐ^
WҐT
JЪG
"К
input_3€€€€€€€€€ДR
!К
input_4€€€€€€€€€
p

 
™ "К€€€€€€€€€д
(__inference_model_1_layer_call_fn_310360Ј8ПО#$/0DACBNKMLРСТУФХЦЧЬЭШЩЪЫЮЯop{|ЗИaҐ^
WҐT
JЪG
"К
input_3€€€€€€€€€ДR
!К
input_4€€€€€€€€€
p 

 
™ "К€€€€€€€€€ж
(__inference_model_1_layer_call_fn_311007є8ПО#$/0CDABMNKLРСТУФХЦЧЬЭШЩЪЫЮЯop{|ЗИcҐ`
YҐV
LЪI
#К 
inputs/0€€€€€€€€€ДR
"К
inputs/1€€€€€€€€€
p

 
™ "К€€€€€€€€€ж
(__inference_model_1_layer_call_fn_311085є8ПО#$/0DACBNKMLРСТУФХЦЧЬЭШЩЪЫЮЯop{|ЗИcҐ`
YҐV
LЪI
#К 
inputs/0€€€€€€€€€ДR
"К
inputs/1€€€€€€€€€
p 

 
™ "К€€€€€€€€€≈
H__inference_sequential_3_layer_call_and_return_conditional_losses_310538yШЩЪЫBҐ?
8Ґ5
+К(
dense_9_input€€€€€€€€€# 
p

 
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ ≈
H__inference_sequential_3_layer_call_and_return_conditional_losses_310552yШЩЪЫBҐ?
8Ґ5
+К(
dense_9_input€€€€€€€€€# 
p 

 
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ Њ
H__inference_sequential_3_layer_call_and_return_conditional_losses_311669rШЩЪЫ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€# 
p

 
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ Њ
H__inference_sequential_3_layer_call_and_return_conditional_losses_311726rШЩЪЫ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€# 
p 

 
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ Э
-__inference_sequential_3_layer_call_fn_310580lШЩЪЫBҐ?
8Ґ5
+К(
dense_9_input€€€€€€€€€# 
p

 
™ "К€€€€€€€€€# Э
-__inference_sequential_3_layer_call_fn_310607lШЩЪЫBҐ?
8Ґ5
+К(
dense_9_input€€€€€€€€€# 
p 

 
™ "К€€€€€€€€€# Ц
-__inference_sequential_3_layer_call_fn_311739eШЩЪЫ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€# 
p

 
™ "К€€€€€€€€€# Ц
-__inference_sequential_3_layer_call_fn_311752eШЩЪЫ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€# 
p 

 
™ "К€€€€€€€€€# Д
$__inference_signature_wrapper_310440џ8ПО#$/0DACBNKMLРСТУФХЦЧЬЭШЩЪЫЮЯop{|ЗИjҐg
Ґ 
`™]
-
input_3"К
input_3€€€€€€€€€ДR
,
input_4!К
input_4€€€€€€€€€"3™0
.
dense_13"К
dense_13€€€€€€€€€љ
Z__inference_token_and_position_embedding_1_layer_call_and_return_conditional_losses_301033_ПО+Ґ(
!Ґ
К
x€€€€€€€€€ДR
™ "*Ґ'
 К
0€€€€€€€€€ДR 
Ъ Х
?__inference_token_and_position_embedding_1_layer_call_fn_303104RПО+Ґ(
!Ґ
К
x€€€€€€€€€ДR
™ "К€€€€€€€€€ДR Џ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_302342Ж РСТУФХЦЧЬЭШЩЪЫЮЯ7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p 
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ Џ
O__inference_transformer_block_3_layer_call_and_return_conditional_losses_302859Ж РСТУФХЦЧЬЭШЩЪЫЮЯ7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p
™ ")Ґ&
К
0€€€€€€€€€# 
Ъ ±
4__inference_transformer_block_3_layer_call_fn_301561y РСТУФХЦЧЬЭШЩЪЫЮЯ7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p 
™ "К€€€€€€€€€# ±
4__inference_transformer_block_3_layer_call_fn_303075y РСТУФХЦЧЬЭШЩЪЫЮЯ7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€# 
p
™ "К€€€€€€€€€# 