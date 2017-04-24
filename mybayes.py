import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        self.training_data = np.asarray(X)
        self.training_labels = np.asarray(y)

        unique_labels = np.unique(self.training_labels)
        unique_feats = np.unique(self.training_data)
        label_count = dict()

        self.feats_count = len(unique_feats)
        self.feat_tag_cmat = np.zeros((len(unique_labels), self.feats_count))
        self.tag_id = {tag: i for i, tag in enumerate(unique_labels)}
        self.feat_id = {feat: i for i, feat in enumerate(unique_feats)}

        for vec, lbl in zip(self.training_data, self.training_labels):
            label_count.setdefault(lbl, 0)
            label_count[lbl] += 1
            for x in vec:
                self.feat_tag_cmat[self.tag_id[lbl]][self.feat_id[x]] += 1

        self.prior_count = label_count
        self.prior_prob = {tag: label_count[tag] / float(len(self.training_labels)) \
                           for tag in unique_labels}

        """
        print "label-count : ",label_count
        print "feat_counts : ", self.feats_count
        print "feat_tag_cmat : ", self.feat_tag_cmat
        print "tag_id : ", self.tag_id
        print "feat_id : ", self.feat_id
        print "prior_count : ", self.prior_count
        print "prior_prob : ", self.prior_prob
        """

    def conditionalProbability(self, val, tag):

        if val in self.feat_id:
            return (self.feat_tag_cmat[self.tag_id[tag]][self.feat_id[val]] + 1.0) / \
                   (self.prior_count[tag] + self.feats_count)
        else:
            return 1.0 / (self.prior_count[tag] + self.feats_count)

    def predict(self, testing_data):

        labels = []
        testing_data = np.asarray(testing_data)

        for i, vec in enumerate(testing_data):
            # initialize smoothed log probabilities for each tag
            smoothed_lp = {tag: 0.0 for tag in self.tag_id}
            for val in vec:
                for tag in self.tag_id:
                    # compute smoothed conditional probability
                    sl_prob = self.conditionalProbability(val, tag)
                    smoothed_lp[tag] += np.log(sl_prob)
                    # Multiply priors
            for tag in self.tag_id:
                smoothed_lp[tag] += self.prior_prob[tag]
            labels.append(max(smoothed_lp.items(), key=lambda x: x[1])[0])

        return labels