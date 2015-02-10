import sys

# Some Models for LDA


class LDATassign():
# Model for topic-word assign model

    def __init__(self, wordmapfilename, tassignfilename):
        print "Loading Model...",
        sys.stdout.flush()
        self.load_wordmap(wordmapfilename)
        self.load_tassign(tassignfilename)
        print "ok"

    def load_wordmap(self, wordmapfilename):
        # Load the word map file
        self.id2word = {}
        with open(wordmapfilename) as f:
            #skip the blank line
            f.readline()
            for l in f:
                word, number = l.strip().split()
                number = int(number)
                self.id2word[number] = word

    def load_tassign(self, tassignfilename):
        #Load the topic-word assignment file
        self.model_topic = {}
        self.model_word = {}
        with open(tassignfilename) as f:
            for l in f:
                words = l.strip().split()
                words = [item.split(':') for item in words]
                for (word_number, topic_number) in words:
                    word_number = int(word_number)
                    topic_number = int(topic_number)
                    if word_number not in self.id2word:
                        continue
                    word = self.id2word[word_number]

                    if word not in self.model_word:
                        self.model_word[word] = {}
                    if topic_number not in self.model_word[word]:
                        self.model_word[word][topic_number] = 0
                    self.model_word[word][topic_number] += 1

                    if topic_number not in self.model_topic:
                        self.model_topic[topic_number] = {}
                    if word not in self.model_topic[topic_number]:
                        self.model_topic[topic_number][word] = 0
                    self.model_topic[topic_number][word] +=1
        #sum the count
        self.word = {}
        for word in self.model_word:
            self.word[word] = sum(self.model_word[word].values())

        self.topic = {}
        for topic in self.model_topic:
            self.topic[topic] = sum(self.model_topic[topic].values())

    def get_cnt(self, word, topic):
        #get the count of (word, topic) pair
        if word not in self.model_word:
            return 0
        if topic not in self.model_word[word]:
            return 0
        return self.model_word[word][topic]

    def get_cntlist_word(self, word):
        #get a list  of (topic, cnt) sorted descendingly
        if word not in self.model_word:
            return None
        return sorted(self.model_word[word].items(), cmp=lambda x, y:-cmp(x[1],y[1]))

    def get_cntlist_topic(self, topic):
        #get a list of (word, cnt) sorted descendingly
        if topic not in self.model_topic:
            return None
        return sorted(self.model_topic[topic].items(), cmp=lambda x,y:-cmp(x[1],y[1]))

    def get_cnt_word(self, word):
        #get the total count of word in Topic Model
        if word not in self.word:
            return 0
        return self.word[word]

    def get_cnt_topic(self, topic):
    #get the totla count of topic in Topic Model
        if topic not in self.topic:
            return 0
        return self.topic[topic]

    #get the probability
    def prob_w_z(self, word, topic):
        #get the probability of p(w|z)
        cnt = self.get_cnt(word, topic)
        cnt_topic = self.get_cnt_topic(topic)
        if cnt_topic == 0:
            return None
        return float(cnt) / float(cnt_topic)

    def prob_z_w(self, word, topic):
        #get the probability of p(z|w)
        cnt = self.get_cnt(word, topic)
        cnt_word = self.get_cnt_word(word)
        if cnt_word == 0:
            return None
        return float(cnt) / float(cnt_word)


if __name__=="__main__":
    m = LDATassign("./../../../CODES/multi_fastLDA/src/wordmap.txt","./../../../CODES/multi_fastLDA/src/model-final.tassign")

