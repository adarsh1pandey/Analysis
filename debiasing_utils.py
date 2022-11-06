import numpy as np
from gensim.models import Word2Vec

class Debiaser(object):
    def __init__(self):
        self.word2vec_model = Word2Vec.load("./word2vec/new_w2v_model.model")
        self.word_vectors = self.word2vec_model.wv
        self.embedding_index = dict()
        with open('pairs.txt', 'r') as f:
            self.lists = f.readlines().split(' ')
            
    def prep_pairs(self):
        self.pairs = []
        for lst in self.lists:
            for i in range(1, len(lst)):
                self.pairs.append([lst[0],lst[i]])

    def prepare_embedding_index(self, x):
        for sent in x:
            for token in sent:
                word = token
                try:
                    coefs = np.asarray(self.word_vectors[word], dtype='float32')
                    self.embeddings_index[word] = coefs
                except:
                    # print(word)
                    coefs = np.zeros((self.embed_size,))
                    self.embeddings_index[word] = coefs

    def cosine_similarity(self, u, v):
        """
        Cosine similarity reflects the degree of similariy between u and v
            
        Arguments:
            u -- a word vector of shape (n,)          
            v -- a word vector of shape (n,)

        Returns:
            cosine_similarity -- the cosine similarity between u and v defined by the formula above.
        """
        
        distance = 0.0
        
        # Compute the dot product between u and v 
        dot = np.dot(u,v)
        # Compute the L2 norm of u 
        norm_u = np.linalg.norm(u)
        
        # Compute the L2 norm of v 
        norm_v = np.linalg.norm(v)
        # Compute the cosine similarity defined by formula (1) 
        cosine_similarity = dot / (norm_u*norm_v)
        
        return cosine_similarity

    def neutralize(self, word, g, word_to_vec_map):
        """
        Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
        This function ensures that gender neutral words are zero in the gender subspace.
        
        Arguments:
            word -- string indicating the word to debias
            g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
            word_to_vec_map -- dictionary mapping words to their corresponding vectors.
        
        Returns:
            e_debiased -- neutralized word vector representation of the input "word"
        """
        
        # Select word vector representation of "word". Use word_to_vec_map. 
        e = word_to_vec_map[word]
        
        # Compute e_biascomponent using the formula give above. 
        e_biascomponent = np.dot(e,g)*g/(np.linalg.norm(g)**2)
    
        # Neutralize e by substracting e_biascomponent from it 
        # e_debiased should be equal to its orthogonal projection. 
        e_debiased = e - e_biascomponent
        
        return e_debiased

    def equalize(self, pair, bias_axis, word_to_vec_map):
        """
        Debias gender specific words by following the equalize method.
        
        Arguments:
        pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
        bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
        word_to_vec_map -- dictionary mapping words to their corresponding vectors
        
        Returns
        e_1 -- word vector corresponding to the first word
        e_2 -- word vector corresponding to the second word
        """
        
        # Step 1: Select word vector representation of "word". Use word_to_vec_map. 
        w1, w2 = pair
        e_w1, e_w2 = word_to_vec_map[w1],word_to_vec_map[w2]
        
        # Step 2: Compute the mean of e_w1 and e_w2
        mu = (e_w1+e_w2)/2

        # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis 
        mu_B = np.dot(mu,bias_axis)*bias_axis/np.linalg.norm(bias_axis)**2
        mu_orth = mu-mu_B

        # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B
        e_w1B = np.dot(e_w1,bias_axis)*bias_axis/np.linalg.norm(bias_axis)**2
        e_w2B = np.dot(e_w2,bias_axis)*bias_axis/np.linalg.norm(bias_axis)**2
            
        # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above
        corrected_e_w1B = np.sqrt(abs(1-np.linalg.norm(mu_orth)**2))*(e_w1B-mu_B)/(abs(e_w1-mu_orth-mu_B))
        corrected_e_w2B = np.sqrt(abs(1-np.linalg.norm(mu_orth)**2))*(e_w2B-mu_B)/(abs(e_w2-mu_orth-mu_B))

        # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections
        e1 = corrected_e_w1B+mu_orth
        e2 = corrected_e_w2B+mu_orth
        
        return e1, e2

    def debias_gender_specific_pairs(self):
        for pair in self.pairs:
            try:
                w1 = pair[0]
                w2 = pair[1]
                g = self.embeddings_index[w2] - self.embeddings_index[w1]
                e1, e2 = self.equalize((w1, w2), g, self.embeddings_index)
                self.embeddings_index[w1] = e1
                self.embeddings_index[w2] = e2
            except:
                pass


