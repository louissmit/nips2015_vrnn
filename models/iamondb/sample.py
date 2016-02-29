import theano
import cPickle as pickle
from cle.cle.utils.op import Gaussian_sample
from cle.cle.utils.gpu_op import concatenate
import theano.tensor as T

theano.config.exception_verbosity='high'

def sample(model_file_name, input_seq):
    train_obj = pickle.load(open(model_file_name, "rb"))
    model = train_obj.model
    nodes = model.nodes
    params = model.params

    [rnn,
     z_1,
     phi_1, phi_mu, phi_sig,
     prior_1, prior_mu, prior_sig,
     theta_1, theta_mu, theta_sig] = nodes

    s_0 = rnn.get_init_state(20)

    x = T.fmatrix('x')

    def inner_fn(x_t, s_tm1):

        phi_1_t = phi_1.fprop([x_t, s_tm1], params)
        phi_mu_t = phi_mu.fprop([phi_1_t], params)
        phi_sig_t = phi_sig.fprop([phi_1_t], params)

        prior_1_t = prior_1.fprop([s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        # z_t = Gaussian_sample(phi_mu_t, phi_sig_t)
        z_1_t = z_1.fprop([phi_mu_t], params)

        s_t = rnn.fprop([[x_t, z_1_t], [s_tm1]], params)

        return s_t, phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, z_1_t


    ((s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp, z_1_temp), updates) = \
        theano.scan(fn=inner_fn,
                    sequences=[x],
                    outputs_info=[s_0, None, None, None, None, None])

    for k, v in updates.iteritems():
        k.default_update = v

    s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)
    theta_1_temp = theta_1.fprop([z_1_temp, s_temp], params)
    theta_mu_temp = theta_mu.fprop([theta_1_temp], params)
    theta_sig_temp = theta_sig.fprop([theta_1_temp], params)

    test_fn = theano.function(inputs=[x],
                              outputs=[phi_mu_temp, phi_sig_temp, theta_mu_temp, theta_sig_temp],
                              updates=updates,
                              allow_input_downcast=True)
    return test_fn(input_seq)
