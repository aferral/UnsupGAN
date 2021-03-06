from infogan.models.regularized_gan import RegularizedGAN
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from infogan.misc.distributions import Bernoulli, Gaussian, Categorical
import sys
import os
import time
from infogan.misc.utils import save_images, inverse_transform, compute_cluster_scores
from sklearn.preprocessing import normalize

TINY = 1e-8
logSamples = 64


class ImprovedGAN(object):
    def __init__(self,
                 model,
                 batch_size=128,
                 dataset=None,
                 val_dataset=None,
                 isTrain=True,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 samples_dir="samples",
                 max_epoch=100,
                 snapshot_interval=500,
                 info_reg_coeff=1.0,
                 discriminator_learning_rate=2e-4,
                 generator_learning_rate=2e-4,
                 semiSup=False
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.samples_dir = samples_dir
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = snapshot_interval
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.info_reg_coeff = info_reg_coeff
        self.discriminator_trainer = None
        self.generator_trainer = None
        self.input_tensor = None
        self.log_vars = []
        self.semiSup = semiSup
        self.input_labels = None

    def init_opt(self):
        if self.dataset.name == "mnist":
            shape = [self.dataset.image_dim]
        elif 'FOLDER' in self.dataset.name:
            print "Selected folder image"
            shape = list(self.dataset.output_size)
        else:
            shape = [self.dataset.output_size, self.dataset.output_size, 3]
        self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size] + shape)

        with pt.defaults_scope(phase=pt.Phase.train):
            self.z_var = self.model.latent_dist.sample_prior(self.batch_size)
            fake_x, _ = self.model.generate(self.z_var)
            self.sample_x, _ = self.model.generate(self.z_var)

            if self.semiSup:
                self.sup_d = self.model.discriminateSup(self.input_tensor, self.dataset.dataObj.getNclasses())
            self.fake_d = self.model.discriminate(fake_x)
            self.real_d = self.model.discriminate(input_tensor)

            self.d_feat_real = self.real_d['features']


            #Define the generator loss as the intermediate layer error (MSE)
            self.d_intermediateLayerGenerated = self.model.calcInterLayer(fake_x)
            self.d_intermediateLayerReal = self.model.calcInterLayer(input_tensor)

            # generator_loss = tf.reduce_mean(tf.squared_difference(self.d_intermediateLayerGenerated, self.d_intermediateLayerReal))
            generator_loss = 0


            if self.semiSup:
                self.input_labels = tf.placeholder(tf.float32, [self.batch_size, self.dataset.dataObj.getNclasses()])
                discriminator_loss_sup = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(self.sup_d['logits'], self.input_labels))

                discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.real_d['logits'],
                                                                                                 tf.zeros_like(
                                                                                                     self.real_d[
                                                                                                         'logits'])))
                discriminator_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(self.fake_d['logits'], tf.ones_like(self.real_d['logits'])))
                discriminator_loss = discriminator_loss_real + discriminator_loss_fake + discriminator_loss_sup

                generator_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.fake_d['logits'],
                                                                                         tf.zeros_like(
                                                                                             self.fake_d['logits'])))

                self.log_vars.append(("discriminator_sup_loss", discriminator_loss_sup))
            else:
                discriminator_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(self.real_d['logits'], tf.ones_like(self.real_d['prob'])))
                discriminator_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(self.fake_d['logits'], tf.zeros_like(self.fake_d['prob'])))
                discriminator_loss = discriminator_loss_real + discriminator_loss_fake
                generator_loss += tf.reduce_mean(
                     tf.nn.sigmoid_cross_entropy_with_logits(self.fake_d['logits'], tf.ones_like(self.fake_d['prob'])))

            self.log_vars.append(("discriminator_loss_real", discriminator_loss_real))
            self.log_vars.append(("discriminator_loss_fake", discriminator_loss_fake))
            self.log_vars.append(("discriminator_loss", discriminator_loss))
            self.log_vars.append(("generator_loss", generator_loss))

            real_d_sum = tf.histogram_summary("real_d", self.real_d['prob'])
            fake_d_sum = tf.histogram_summary("fake_d", self.fake_d['prob'])

            if self.model.is_reg:
                reg_z = self.model.reg_z(self.z_var)
                mi_est = tf.constant(0.)
                cross_ent = tf.constant(0.)

                # compute for discrete and continuous codes separately
                # discrete:
                if len(self.model.reg_disc_latent_dist.dists) > 0:
                    disc_reg_z = self.model.disc_reg_z(reg_z)
                    disc_reg_dist_info = self.model.disc_reg_dist_info(
                        self.fake_d['reg_dist_info'])  # Returns a dictionary of activations for each distribution
                    disc_log_q_c_given_x = self.model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)
                    disc_log_q_c = self.model.reg_disc_latent_dist.logli_prior(disc_reg_z)
                    disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
                    disc_ent = tf.reduce_mean(-disc_log_q_c)
                    disc_mi_est = disc_ent - disc_cross_ent
                    mi_est += disc_mi_est
                    cross_ent += disc_cross_ent
                    self.log_vars.append(("MI_disc", disc_mi_est))
                    self.log_vars.append(("CrossEnt_disc", disc_cross_ent))
                    discriminator_loss -= self.info_reg_coeff * disc_mi_est
                    generator_loss -= self.info_reg_coeff * disc_mi_est

                    real_disc_reg_dist_info = self.model.disc_reg_dist_info(self.real_d['reg_dist_info'])
                    assert len(
                        real_disc_reg_dist_info.keys()) == 1  # currently support only one categorical distribution
                    self.disc_prob = real_disc_reg_dist_info[real_disc_reg_dist_info.keys()[0]]

                if len(self.model.reg_cont_latent_dist.dists) > 0:
                    cont_reg_z = self.model.cont_reg_z(reg_z)
                    cont_reg_dist_info = self.model.cont_reg_dist_info(self.fake_d['reg_dist_info'])
                    cont_log_q_c_given_x = self.model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)
                    cont_log_q_c = self.model.reg_cont_latent_dist.logli_prior(cont_reg_z)
                    cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
                    cont_ent = tf.reduce_mean(-cont_log_q_c)
                    cont_mi_est = cont_ent - cont_cross_ent
                    mi_est += cont_mi_est
                    cross_ent += cont_cross_ent
                    self.log_vars.append(("MI_cont", cont_mi_est))
                    self.log_vars.append(("CrossEnt_cont", cont_cross_ent))
                    discriminator_loss -= self.info_reg_coeff * cont_mi_est
                    generator_loss -= self.info_reg_coeff * cont_mi_est

                for idx, dist_info in enumerate(
                        self.model.reg_latent_dist.split_dist_info(self.fake_d['reg_dist_info'])):
                    if "stddev" in dist_info:
                        self.log_vars.append(("max_std_%d" % idx, tf.reduce_max(dist_info["stddev"])))
                        self.log_vars.append(("min_std_%d" % idx, tf.reduce_min(dist_info["stddev"])))

                self.log_vars.append(("MI", mi_est))
                self.log_vars.append(("CrossEnt", cross_ent))

            all_vars = tf.trainable_variables()
            d_vars = [var for var in all_vars if var.name.startswith('d_')]
            g_vars = [var for var in all_vars if var.name.startswith('g_')]

            discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
            self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer, losses=[discriminator_loss],
                                                            var_list=d_vars)

            generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
            self.generator_trainer = pt.apply_optimizer(generator_optimizer, losses=[generator_loss], var_list=g_vars)

            for k, v in self.log_vars:
                tf.scalar_summary(k, v)

        if self.model.is_reg and self.dataset.name != 'imagenet':
            if self.model.encoder_dim <= 12:  # Ugly conditioning!!! Fix later
                with pt.defaults_scope(phase=pt.Phase.test):
                    with tf.variable_scope("model", reuse=True) as scope:
                        self.visualize_all_factors()

    def visualize_all_factors(self):
        with tf.Session():
            fixed_noncat = np.concatenate([
                np.tile(
                    self.model.nonreg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.nonreg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)
            fixed_cat = np.concatenate([
                np.tile(
                    self.model.reg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.reg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)

        offset = 0
        for dist_idx, dist in enumerate(self.model.reg_latent_dist.dists):
            if isinstance(dist, Gaussian):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                c_vals = []
                for idx in xrange(10):
                    c_vals.extend([-1.0 + idx * 2.0 / 9] * 10)
                c_vals.extend([0.] * (self.batch_size - 100))
                vary_cat = np.asarray(c_vals, dtype=np.float32).reshape((-1, 1))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset + 1] = vary_cat
                offset += 1
            elif isinstance(dist, Categorical):
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(dist.dim):
                    cat_ids.extend([idx] * 10)
                cat_ids.extend([0] * (self.batch_size - 10 * dist.dim))
                print cat_ids
                cur_cat = np.copy(fixed_cat)
                print cur_cat.shape
                cur_cat[:, offset:offset + dist.dim] = lookup[cat_ids]
                offset += dist.dim
            elif isinstance(dist, Bernoulli):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([int(idx / 5)] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset + dist.dim] = np.expand_dims(np.array(cat_ids), axis=-1)
                # import ipdb; ipdb.set_trace()
                offset += dist.dim
            else:
                raise NotImplementedError
            z_var = tf.constant(np.concatenate([fixed_noncat, cur_cat], axis=1))

            x_dist_flat, x_dist_info = self.model.generate(z_var)

            # just take the mean image
            transform_type = 'output_dist'
            if isinstance(self.model.output_dist, Bernoulli):
                img_var = x_dist_info["p"]
            elif isinstance(self.model.output_dist, Gaussian):
                img_var = x_dist_info["mean"]
            elif self.model.output_dist is None:
                img_var = x_dist_flat
                transform_type = None
            else:
                raise NotImplementedError
            if transform_type == 'output_dist':
                img_var = self.dataset.inverse_transform(img_var)
            else:
                img_var = inverse_transform(img_var)
            rows = dist.dim
            cols = 10
            img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
            img_var = img_var[:rows * cols, :, :, :]
            imgs = tf.reshape(img_var, [rows, cols] + list(self.dataset.image_shape))
            stacked_img = []
            for row in xrange(rows):
                row_img = []
                for col in xrange(cols):
                    row_img.append(imgs[row, col, :, :, :])
                stacked_img.append(tf.concat(1, row_img))
            imgs = tf.concat(0, stacked_img)
            imgs = tf.expand_dims(imgs, 0)
            tf.image_summary("image_%d_%s" % (dist_idx, dist.__class__.__name__), imgs)

    def getFeedDict(self, dataset, semiSup):
        x, labels = dataset.next_batch(self.batch_size)

        if semiSup:
            feed_dict = {self.input_tensor: x, self.input_labels: labels}
        else:
            feed_dict = {self.input_tensor: x}
        return feed_dict

    def train(self, sess):
        init = tf.initialize_all_variables()
        sess.run(init)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)
        saver = tf.train.Saver()

        counter = 0

        log_vars = [x for _, x in self.log_vars]
        log_keys = [x for x, _ in self.log_vars]

        start_time = time.time()

        for epoch in range(self.max_epoch):
            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            print("DatasetName ", self.dataset.name)
            print('batchIdx ', self.dataset.batch_idx)
            pbar = ProgressBar(maxval=self.dataset.batch_idx['train'], widgets=widgets)
            pbar.start()

            all_log_vals = []
            for i in range(self.dataset.batch_idx['train']):
                pbar.update(i)

                feed_dict = self.getFeedDict(self.dataset, self.semiSup)

                all_log_vals = sess.run([self.discriminator_trainer] + log_vars, feed_dict)[1:]
                for ii in range(5):
                    sess.run(self.generator_trainer, feed_dict)
                counter += 1

                if (counter % self.snapshot_interval == 0):
                    snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                    fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                    print("Model saved in file: %s" % fn)

                # Save samples
                if counter % 10 == 0:
                    samples = sess.run(self.sample_x, feed_dict)
                    samples = samples[:logSamples, ...]
                    sqS = int(np.sqrt(logSamples))
                    if self.dataset.name != "mnist":
                        samples = inverse_transform(samples)
                        # xTolog = inverse_transform(x)[:logSamples, ...]
                        # save_images(xTolog, [sqS, sqS],'{}/trainDATA_{:02d}_{:04d}.png'.format(self.samples_dir, epoch, counter))

                    save_images(samples, [sqS, sqS],
                                '{}/train_{:02d}_{:04d}.png'.format(self.samples_dir, epoch, counter))

                # Test on validation (test) set
                if counter % 500 == 1:
                    print "Validating current model on val set..."
                    self.validate(sess)

                # Get next batch
                # feed_dict = self.getFeedDict(self.dataset, self.semiSup)

                # Write summary to log file

                summary_str = sess.run(summary_op, feed_dict)
                summary_writer.add_summary(summary_str, counter)

                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, all_log_vals))
                print("Epoch %d | time: %4.4fs " % (epoch, time.time() - start_time) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(all_log_vals)):
                    raise ValueError("NaN detected!")
        print "Train ended"
        # Test on validation (test) set
        snapshot_name = "last"
        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
        print("Model saved in file: %s" % fn)

    def validate(self, sess):
        pred_labels = np.array([], dtype=np.int16).reshape(0, )
        pred_labels_kmeans = np.array([], dtype=np.int16).reshape(0, )
        labels = []
        n_clusters = self.val_dataset.n_labels
        if self.model.is_reg is True and self.model.encoder_dim == n_clusters:
            predict_directly = True
        else:
            predict_directly = False

        trainX = np.array([]).reshape(0, 0)

        def pool_features(feat, pool_type='avg'):
            if len(feat.shape) >= 3:
                if pool_type == 'avg':
                    feat = feat.mean(axis=(1, 2))
                if pool_type == 'max':
                    feat = feat.mean(axis=(1, 2))
            return feat.reshape((feat.shape[0], feat.shape[-1]))

        print "Getting all the training features."
        for ii in range(self.val_dataset.batch_idx['train']):

            feed_dict = self.getFeedDict(self.dataset, self.semiSup)

            d_features = sess.run(self.d_feat_real, feed_dict)
            d_features = pool_features(d_features, pool_type='avg')
            if trainX.shape[0] == 0:  # Is empty
                trainX = d_features
            else:
                trainX = np.concatenate((trainX, d_features), axis=0)
        trainX_norm = normalize(trainX, axis=1, norm='l2')
        print "Learning the clusters."
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(trainX_norm)
        print "Extracting features from val set and predicting from it."
        for ii in range(self.val_dataset.batch_idx['val']):
            x, batch_labels = self.val_dataset.next_batch(self.batch_size, split="val")

            if type(batch_labels) == np.ndarray:
                batch_labels = list(batch_labels)
            labels = labels + batch_labels

            if predict_directly:
                d_prob = sess.run(self.disc_prob, {self.input_tensor: x})
                batch_pred_labels = np.argmax(d_prob, axis=1)
                pred_labels = np.concatenate((pred_labels, batch_pred_labels))

            d_features = sess.run(self.d_feat_real, {self.input_tensor: x})
            d_features = pool_features(d_features, pool_type='avg')
            d_features_norm = normalize(d_features, axis=1, norm='l2')
            batch_pred_labels_kmeans = kmeans.predict(d_features_norm)

            pred_labels_kmeans = np.concatenate((pred_labels_kmeans, batch_pred_labels_kmeans))

        if predict_directly:
            compute_cluster_scores(labels=np.asarray(labels), pred_labels=pred_labels,
                                   path=os.path.join(self.log_dir, 'scores.txt'))

        compute_cluster_scores(labels=np.asarray(labels), pred_labels=pred_labels_kmeans,
                               path=os.path.join(self.log_dir, 'scores_kmeans.txt'))
