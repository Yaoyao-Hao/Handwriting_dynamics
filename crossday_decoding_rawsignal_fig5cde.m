%% cross-day decoding using raw neural signals fig5ce
clc
clear all

sessions = {'0614','0616','0623','0624','0630','0701'};
session_num=1:6;

acc = cell(5,1);
for tr_session_num=1:5 % num of training sessions
    combs = nchoosek(session_num, tr_session_num);   % all combinations
    acc_session = [];
    for comb=1:size(combs,1)
        disp(combs(comb,:))
        tr_labels = [];
        data_tr = [];
        for session=1:size(combs(comb,:),2)
            disp(['train session: ' sessions{combs(comb,session)}])
            data = load([sessions{combs(comb,session)} '_mua_noaver_smo.mat']);
            stroke = data.Stroke_cell;
            penlift = data.Penlift_cell;
           
            stroke_data = cellfun(@(x) struct('matrix', x), stroke);
            penlift_data = cellfun(@(x) struct('matrix', x), penlift);
            for i = 1:length(stroke_data)
                data_tr = [data_tr; stroke_data(i).matrix(:)'];
            end
            for i = 1:length(penlift_data)
                data_tr = [data_tr; penlift_data(i).matrix(:)'];
            end

            tmp_labels = [zeros(size(stroke,1),1);ones(size(penlift,1),1)];
            tr_labels = [tr_labels;tmp_labels];
        end
        model = fitcsvm(data_tr, tr_labels, 'KernelFunction', 'linear'); % training a svm

        test_sessions = setdiff(session_num, combs(comb,:)); % test sessions
        disp(test_sessions)
        for session=1:size(test_sessions,2)
            disp(['test session: ' sessions{test_sessions(session)}])
            data_tt = [];
            data = load([sessions{test_sessions(session)} '_mua_noaver_smo.mat']);
            stroke = data.Stroke_cell;
            penlift = data.Penlift_cell; 

            stroke_data = cellfun(@(x) struct('matrix', x), stroke);
            penlift_data = cellfun(@(x) struct('matrix', x), penlift);
            for i = 1:length(stroke_data)
                data_tt = [data_tt; stroke_data(i).matrix(:)'];
            end
            for i = 1:length(penlift_data)
                data_tt = [data_tt; penlift_data(i).matrix(:)'];
            end

            tt_labels = [zeros(size(stroke,1),1);ones(size(penlift,1),1)];

            y_pred = predict(model, data_tt);
            acc_session = [acc_session sum(y_pred == tt_labels) / length(tt_labels)];
        end
    end
    acc{tr_session_num} = acc_session;
end
