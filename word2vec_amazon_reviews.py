# Uninstall pre-installed packages that may cause version conflicts
!pip uninstall -y gensim scipy numpy

# Reinstall compatible versions of gensim, scipy, and numpy
!pip install numpy==1.23.5
!pip install scipy==1.10.1
!pip install gensim==4.3.1
#----------------------------------------------------------------------

# Import necessary libraries
import gensim
import pandas as pd
#------------------------------------------------------------------------

# Mount Google Drive to access dataset stored in Drive
from google.colab import drive
drive.mount('/content/drive')
#--------------------------------------------------------------------

# Load the Amazon Cell Phones and Accessories reviews dataset (JSON format)
df=pd.read_json('/path/Cell_Phones_and_Accessories_5.json',lines=True)
#----------------------------------------------------------------------

# Preview the dataset
df.head()
#----------------------------------------------------------------------

# Preprocess the review text:
# - Tokenize
# - Lowercase
# - Remove punctuation and stopwords
review_text=df.reviewText.apply(gensim.utils.simple_preprocess)
review_text
#----------------------------------------------------------------------------

# Initialize the Word2Vec model
# - window: context window size
# - min_count: ignore words with total frequency lower than this
# - workers: number of threads
model=gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4
)
#---------------------------------------------------------------------------

# Build the vocabulary from the preprocessed reviews
model.build_vocab(review_text,progress_per=1000)
#-----------------------------------------------------------------------

# Train the Word2Vec model on the review data
model.train(review_text,total_examples=model.corpus_count,epochs=model.epochs)
#-------------------------------------------------------------------------------

# Find words most similar to 'bad' using the trained embeddings
model.wv.most_similar('bad')
#-----------------------------------------------------------------------------

# Calculate similarity between 'good' and 'great'
model.wv.similarity(w1='good',w2='great')
