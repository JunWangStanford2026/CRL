import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_greedy(observation, model):
    '''
    observation is a length-N sequence containing tokens in {0, 1, ..., N - 1}
    'N' is a special token used for padding
    model is a PyTorch model whose output has shape (1, N), containing the predicted probabilities of the next token over {0, 1, ..., N - 1}
    returns a tuple (response, log probability of response)
    '''

    # prepend the observation with padding of N's to ensure context always contains full observation
    N = len(observation)
    context = [N] * N + list(observation)

    # generate response autoregressively
    # at each time shift context window by 1
    log_prob = torch.zeros(N, dtype=float).to(device)
    for i in range(N):
        input = torch.LongTensor(context).unsqueeze(0).to(device)  # shape (1, 2N)
        output = model(input)  # shape (1, N)
        predicted_token = torch.argmax(output, dim=-1).cpu().item()  # shape (1,)
        assert type(predicted_token) == int
        log_prob[i] = torch.log(output[0, predicted_token])  # record log probability
        context = context[1:] + [predicted_token]  # shift context window by 1

    return np.array(context[-N:]), log_prob  # return the generated response and its token-wise log probability

# Deprecated
# def generate_thompson_sampling(observation, model):
#     '''
#         same as generate_greedy() but samples from the model's output distribution instead of taking the argmax
#     '''
#     N = len(observation)
#     context = [N] * N + list(observation)
#     log_prob = torch.zeros(N, dtype=float).to(device)
#     for i in range(N):
#         input = torch.LongTensor(context).unsqueeze(0).to(device)  # shape (1, 2N)
#         output = model(input)  # shape (1, N)
#         distribution = torch.distributions.Categorical(probs=output[0])
#         predicted_token = distribution.sample().item()  # sample from the distribution
#         assert type(predicted_token) == int
#         log_prob[i] = distribution.log_prob(torch.tensor(predicted_token).to(device))  # accumulate log probability
#         context = context[1:] + [predicted_token]  # shift context window by 1
#     return np.array(context[-N:]), log_prob  # return the generated response and its log probability


def generate(observation, model, batch_size=16):
    '''
        generate action from the model's posterior
    '''
    N = len(observation)
    context = np.tile(np.array([N] * N + list(observation)), (batch_size, 1)) # (B, 2 * N)
    assert context.shape == (batch_size, 2 * N)
    log_prob = torch.zeros((batch_size, N), dtype=float).to(device)
    for i in range(N):
        input = torch.LongTensor(context).to(device)  # shape (B, 2N)
        output = model(input)  # shape (B, N)
        distribution = torch.distributions.Categorical(output)
        predicted_tokens = distribution.sample()  # sample from the distribution, size (B,)
        assert predicted_tokens.shape == (batch_size,)
        log_prob[:, i] = distribution.log_prob(predicted_tokens)
        context = np.concatenate((context[:, 1:], predicted_tokens.cpu().numpy()[:, None]), axis=1)  # shift context window by 1
        assert context.shape == (batch_size, 2 * N)
    return np.array(context[:, -N:]), log_prob  # return the generated response (B, N) and their log probabilities (B,)


def generate_greedy_hra(observation, model, num_output_heads):
    '''
    observation is a length-N sequence containing tokens in {0, 1, ..., N - 1}
    'N' is a special token used for padding
    model is a PyTorch model whose output has shape (1, num_output_heads, N), containing the predicted probabilities of various heads for the next token over {0, 1, ..., N - 1}
    returns a tuple (response, log probability of response)
    '''

    # prepend the observation with padding of N's to ensure context always contains full observation
    N = len(observation)
    context = [N] * N + list(observation)

    # generate response autoregressively
    # at each time shift context window by 1
    log_prob = torch.zeros((num_output_heads, N), dtype=float, requires_grad=False).to(device)
    for i in range(N):
        input = torch.LongTensor(context).unsqueeze(0).to(device)  # shape (1, 2N)
        output_heads = model(input)  # shape (1, num_output_heads, N)
        output_combined = torch.sum(output_heads, dim=1).squeeze(0) # shape(1, N) -> (N,)
        predicted_token = torch.argmax(output_combined)
        log_prob[:, i] = torch.log(output_heads[0, :, predicted_token])  # record log probability
        context = context[1:] + [predicted_token]  # shift context window by 1

    return np.array(context[-N:]), log_prob  # return the generated response and its token-wise log probability


def generate_hra(observation, model, num_output_heads, batch_size=32, epsilon=1e-6):
    '''
    model outputs must be normalized probabilities that sum to 1
    '''

    # prepend the observation with padding of N's to ensure context always contains full observation
    N = len(observation)
    context = np.tile(np.array([N] * N + list(observation)), (batch_size, 1)) # (B, 2 * N)
    log_prob = torch.zeros((batch_size, num_output_heads, N), dtype=float).to(device)

    for i in range(N):
        input = torch.LongTensor(context).to(device) # shape (B, 2N)
        output_heads = model(input) # (B, num_output_heads, N)
        assert output_heads.shape[1] == num_output_heads
        output_combined = torch.mean(output_heads, dim=1) # (B, N)
        distribution = torch.distributions.Categorical(output_combined)
        predicted_tokens = distribution.sample() # size (B,)
        assert predicted_tokens.shape == (batch_size,)
        # for each of B samples, log probs of the selected token based on the probabilities from the different heads, shape (B, num_output_heads)
        # the token is sampled from the mean over heads, so an individual head can assign it ~0 probability;
        # clamp before the log, since log's backward pass (1/p) overflows float32 once p reaches the denormal
        # range (~1e-38) and turns every gradient into NaN. clamped entries pass zero gradient.
        log_prob[:, :, i] = torch.log(output_heads.gather(
                                        -1, predicted_tokens.view(batch_size, 1, 1).expand(-1, num_output_heads, -1)
                                    ).squeeze(-1).clamp_min(epsilon))
        context = np.concatenate((context[:, 1:], predicted_tokens.cpu().numpy()[:, None]), axis=1) # shift context window by 1
        assert context.shape == (batch_size, 2 * N)

    return np.array(context[:, -N:]), log_prob



def generate_compositional(observation, model_0, model_1):
    '''
    model outputs must be normalized probabilities that sum to 1
    '''

    # prepend the observation with padding of N's to ensure context always contains full observation
    N = len(observation)
    context = [N] * N + list(observation)

    # generate response autoregressively
    # at each time shift context window by 1
    for i in range(N):
        input = torch.LongTensor(context).unsqueeze(0).to(device)  # shape (1, 2N)
        output_0, output_1 = model_0(input), model_1(input)  # shape (1, N)
        output = torch.log(output_0) + torch.log(output_1)
        predicted_token = torch.argmax(output, dim=-1).cpu().item()  # shape (1,)
        assert type(predicted_token) == int
        context = context[1:] + [predicted_token]  # shift context window by 1

    return np.array(context[-N:])  # return the generated response


