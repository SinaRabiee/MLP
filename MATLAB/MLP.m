function [W , V] = MLP(L,D,Z,P,J,K)
    C = 4.5;
    Emax = 0.1;
    E = 0;
    V = ones(J-1,L);
    W = ones(K,J);
    
%     V = rand(J-1,L)*2;
%     W = rand(K,J)*2;
    Z = cat(1,Z,ones(1,P));
    Y = ones(1,J);
    delta_Y = zeros(1,J-1);
    O = zeros(1,K);
    delta_O = zeros(1,K);
    
    while (true)
        for p = 1:P
            % 2nd layer
            for k = 1:K
                O(k) = 2 / (1+exp(-W(k,:) *  Y')) - 1;
            end

            % 1st later
            for j = 1:J-1
                Y(j) = 2 / (1+exp(-V(j,:) *  Z(:,p))) - 1;
            end

            E = E + norm(D(:,p)-O,2)/2;

            % output error signal
            for k = 1:K
                delta_O(K) = (D(k,p) - O(k)) * (1 - O(k)^2)/2;
            end
            
            % hidden layer error signal
            for j = 1:J-1
                delta_Y(j) = (delta_O * W(:,j)) * (1 - Y(j)^2)/2;
            end

            % updating weights
            for k = 1:K
                W(k,:) = W(k,:) + C*delta_O(k)*Y;
            end

            for i = 1:L
                V = V +  C*(Z(:,p)*delta_Y)';
            end
        end

        if (E < Emax)
            break
        else 
            disp(E)
            E = 0;
        end
    end 
end