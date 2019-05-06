#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <HFO.hpp>
#include <xgboost/c_api.h>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <bits/stdc++.h>

using namespace hfo;

#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  exit(1);                                                              \
}                                                                       \
}

typedef std::vector<float> State;

float pitchHalfLength = 52.5;
float pitchHalfWidth = 34;
float tolerance_x = pitchHalfLength * 0.1;
float tolerance_y = pitchHalfWidth * 0.1;
float FEAT_MIN = -1;
float FEAT_MAX = 1;
float pi = 3.14159265358979323846;
float max_R = sqrtf(pitchHalfLength * pitchHalfLength + pitchHalfWidth * pitchHalfWidth);
float stamina_max = 8000;
feature_set_t features = HIGH_LEVEL_FEATURE_SET;
std::string config_dir = "bin/teams/base/config/formations-dt";
int port = 6000;
std::string server_addr = "localhost";
std::string team_name = "base_right";
bool goalie = false;

float unnormalize(float val, float min_val, float max_val)
{
    if (val < FEAT_MIN || val > FEAT_MAX)
    {
        std::cout << "Unnormalized value Violated Feature Bounds: " << val
                  << " Expected min/max: [" << FEAT_MIN << ", "
                  << FEAT_MAX << "]" << std::endl;
        float ft_max = FEAT_MAX; // Linker error on OSX otherwise...?
        float ft_min = FEAT_MIN;
        val = std::min(std::max(val, ft_min), ft_max);
    }
    return ((val - FEAT_MIN) / (FEAT_MAX - FEAT_MIN)) * (max_val - min_val) + min_val;
}

float abs_x(float normalized_x_pos, bool playingOffense)
{
    float tolerance_x = .1 * pitchHalfLength;
    if (playingOffense)
    {
        return unnormalize(normalized_x_pos, -tolerance_x, pitchHalfLength + tolerance_x);
    }
    else
    {
        return unnormalize(normalized_x_pos, -pitchHalfLength - tolerance_x, tolerance_x);
    }
}

float abs_y(float normalized_y_pos)
{
    float tolerance_y = .1 * pitchHalfWidth;
    return unnormalize(normalized_y_pos, -pitchHalfWidth - tolerance_y,
                       pitchHalfWidth + tolerance_y);
}

void remake_state(State &state, int num_mates, int num_ops, bool is_offensive)
{
    state[0] = abs_x(state[0], is_offensive);
    state[1] = abs_y(state[1]);
    state[2] = unnormalize(state[2], -pi, pi);
    state[3] = abs_x(state[3], is_offensive);
    state[4] = abs_y(state[4]);
    state[5] = unnormalize(state[5], 0, 1);
    state[6] = unnormalize(state[6], 0, max_R);
    state[7] = unnormalize(state[7], -pi, pi);
    state[8] = unnormalize(state[8], 0, pi);
    if (num_ops > 0)
        state[9] = unnormalize(state[9], 0, max_R);
    else
        state[9] = -1000;
    for (int i = 10; i < 10 + num_mates; i++)
    {
        if (state[i] != -2)
            state[i] = unnormalize(state[i], 0, pi);
        else
            state[i] = -1000;
    }
    for (int i = 10 + num_mates; i < 10 + 2 * num_mates; i++)
    {
        if (state[i] != -2)
            state[i] = unnormalize(state[i], 0, max_R);
        else
            state[i] = -1000;
    }
    for (int i = 10 + 2 * num_mates; i < 10 + 3 * num_mates; i++)
    {
        if (state[i] != -2)
            state[i] = unnormalize(state[i], 0, pi);
        else
            state[i] = -1000;
    }
    int index = 10 + 3 * num_mates;
    for (int i = 0; i < num_mates; i++)
    {
        if (state[index] != -2)
            state[index] = abs_x(state[index], is_offensive);
        else
            state[index] = -1000;
        index += 1;
        if (state[index] != -2)
            state[index] = abs_y(state[index]);
        else
            state[index] = -1000;
        index += 2;
    }
    index = 10 + 6 * num_mates;
    for (int i = 0; i < num_ops; i++)
    {
        if (state[index] != -2)
            state[index] = abs_x(state[index], is_offensive);
        else
            state[index] = -1000;
        index += 1;
        if (state[index] != -2)
            state[index] = abs_y(state[index]);
        else
            state[index] = -1000;
        index += 2;
    }
    state[state.size() - 1] = unnormalize(state[state.size() - 1], 0, stamina_max);
}

State strict_state(State &state, int num_mates, int choosed_mates = 6, int choosed_ops = 7)
{
    State new_state;
    for (int i = 0; i < 10; i++)
        new_state.push_back(state[i]);
    for (int i = 10; i < 10 + choosed_mates; i++)
        new_state.push_back(state[i]);
    for (int i = 10 + num_mates; i < 10 + num_mates + choosed_mates; i++)
        new_state.push_back(state[i]);
    for (int i = 10 + (2*num_mates); i < 10 + (2*num_mates) + choosed_mates; i++)
        new_state.push_back(state[i]);
    int index = 10 + (3*num_mates);
    for (int i = 0; i < choosed_mates; i++)
    {
        new_state.push_back(state[index]);
        index += 1;
        new_state.push_back(state[index]);
        index += 1;
        new_state.push_back(state[index]);
        index += 1;
    }
    index = 10 + (6*num_mates);
    for (int i = 0; i < choosed_ops; i++)
    {
        new_state.push_back(state[index]);
        index += 1;
        new_state.push_back(state[index]);
        index += 1;
        new_state.push_back(state[index]);
        index += 1;
    }
    int len = state.size();
    new_state.push_back(state[len - 1]);
    new_state.push_back(state[len - 2]);
    return new_state;
}

int main(int argc, char **argv)
{
    int silent = 0;
    BoosterHandle booster;
    safe_xgboost(XGBoosterCreate(NULL, 0, &booster))
    HFOEnvironment hfo;
    // Connect to the server and request feature set. See manual for
    // more information on feature sets.
    hfo.connectToServer(features, config_dir, port, server_addr,
                        team_name, goalie);
    int num_mates = hfo.getNumTeammates();
    int num_ops = hfo.getNumOpponents();
    std::vector<hfo::action_t> actions = {MOVE, GO_TO_BALL};
    int unum = hfo.getUnum();

    std::string fname_str = std::string("agent_") + std::to_string(unum) + std::string(".model");
    int n = fname_str.size();
    char fname[n + 1];
    std::strcpy(fname, fname_str.c_str());
    safe_xgboost(XGBoosterLoadModel(booster, fname))

    status_t status = IN_GAME;
    for (int episode = 0; status != SERVER_DOWN; episode++)
    {
        status = IN_GAME;
        while (status == IN_GAME)
        {
            const State &hfo_state = hfo.getState();
            State state = hfo_state;
            remake_state(state, num_mates, num_ops, false);
            State remake = strict_state(state, num_mates);
            DMatrixHandle d_state;
            float my_state[remake.size()];
            for (int i=0; i < remake.size(); i++)
            {
                my_state[i] = remake[i];
            }
            safe_xgboost(XGDMatrixCreateFromMat(my_state, 1, remake.size(), NAN, &d_state))
            bst_ulong out_len;
            const float *result;
            safe_xgboost(XGBoosterPredict(booster, d_state, 0, 0, &out_len, &result))
            int act = (result[0] < 0.5) ? 0 : 1;
            hfo.act(actions[act]);
            hfo.step();
        }
    }
    hfo.act(QUIT);
}