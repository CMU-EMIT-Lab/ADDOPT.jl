@time using Plots
@time using ProgressMeter

function animate_state_history(X, dt, n_rows, n_cols; path="animation_state.mp4", strid=1, width=1000, height=1000)
    N = n_rows * n_cols

    p = Progress(length(X) ÷ strid)
    anim = @animate for k in 1:strid:length(X)
        Ek = view(X[k],1:N)
        mk = view(X[k], (N+1):2N)
        # yk = view(X[k], (2N+1):3N)
        E_arr = reshape(Ek, (n_cols, n_rows))'
        m_arr = reshape(mk, (n_cols, n_rows))'
        # y_arr = reshape(yk, (n_cols, n_rows))'

        hm_E = heatmap(1:n_cols, 1:n_rows, E_arr, aspect_ratio=:equal, clim=(0, 100), title="Internal Energy (J)")
        hm_m = heatmap(1:n_cols, 1:n_rows, m_arr, aspect_ratio=:equal, clim=(0, 1), title="Mass (mg)")
        # hm_y = heatmap(1:n_cols, 1:n_rows, y_arr, aspect_ratio=:equal, clim=(0, 1.5), title="Fraction Transformed")
        # plot(hm_E, hm_m, hm_y, layout = (3, 1), size=(width, height), fmt=:png)
        plot(hm_E, hm_m, layout = (2, 1), size=(width, height), fmt=:png)

        
        next!(p)
    end
    gif(anim, path, fps=(1/dt/strid))
    
end

function animate_measurement_history(Y, dt, n_rows, n_cols; path="animation_measured.mp4", strid=1, width=1000, height=400)
    N = n_rows * n_cols
    
    p = Progress(length(Y) ÷ strid)
    anim = @animate for k in 1:strid:length(Y)
        Tk = Y[k]
        T_arr = reshape(Tk, (n_cols, n_rows))'

        hm_T = heatmap(1:n_cols, 1:n_rows, T_arr, aspect_ratio=:equal, clim=(300, 1600), title="Measured Temperature K")
        plot(hm_T, size=(width, height), fmt=:png)
        
        next!(p)
    end
    gif(anim, path, fps=(1/dt/strid))
    
end