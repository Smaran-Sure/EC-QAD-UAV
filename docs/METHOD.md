# Method

## Scope and hypotheses

EC-QAD studies a training interface between physical state estimation and policy compression. H1 asks whether estimator statistics identify compression-critical conditions beyond fault status and generic weighting. H2 asks whether weighting the distillation loss with those statistics improves closed-loop flight. They are separate hypotheses. Covariance is modeled uncertainty, NIS is consistency information, neither is a calibrated risk or fault probability. Fixed-bit weighting redistributes training emphasis; it does not dynamically allocate bits.

## Aircraft and conventional controller

JSBSim nonlinear six-degree-of-freedom Rascal110: ArduPilot source revision `2a54162249a9b02845d5b62920d29d5edaece82e`, vendored XML and propulsion files in `models/jsbsim`. The aerodynamic parameters are a public BETA approximation, not independently validated aerodynamic identification. Model dimensions include wing area 10.57 ft², span 9.17 ft, empty weight 13 lb plus 1.5 lb tank contents. The nominal initial altitude is 300 m and calibrated airspeed 25 m/s; true airspeed differs with density. The simulated aircraft is a small remotely piloted fixed-wing platform, not a C172 substitute.

Dynamics run at 200 Hz, controller at 50 Hz. Conventional control cascades heading→roll→aileron, altitude→pitch→elevator, airspeed→throttle, plus rudder yaw-rate damping. Exact gains/limits are in `configs/controller.yaml`. Reference steps occur at 30, 60, 90, 120, and 150 s; heading, altitude, and airspeed changes are randomized jointly in scale and timing. A fixed initial reference is established from the prescribed initial condition before feedback; runtime feedback uses only estimated navigation and measurements. No truth-state fallback is allowed.

Built-in trim was abandoned after the existing failed trim investigation. The configured initial controls and autopilot establish flight. A common actuator startup bound `throttle_actual <= 1e-5 + t/2` avoids the documented zero-RPM electric motor impulse (JSBSim issue 1269). `set-running` is disabled for this electric model because its steady-state initializer uses a large step. Aerodynamic and engine parameters were not altered. Startup is included in all episode metrics.

## Sensors and UKF

Sensors are generated from truth, but truth is not supplied to the learned policies or estimator. Gyroscope and specific-force accelerometer: 100 Hz; GPS position/velocity: 5 Hz; heading observation and airspeed: 20 Hz. Independent Gaussian measurement noises and episode-constant random biases are configured in `configs/sensors.yaml`. Specific force is generated from finite-difference NED acceleration, rotated to body coordinates, with gravity removed. This is a simplified sensor simulation, not a calibrated hardware sensor suite. Heading is a direct angular observation, not a full magnetic-field model. Still air is assumed; no wind estimator is claimed.

The 15-state navigation UKF estimates `[p_N,p_E,p_D,v_N,v_E,v_D,roll,pitch,yaw,b_gx,b_gy,b_gz,b_ax,b_ay,b_az]`. Strapdown prediction integrates bias-corrected gyroscope and specific force. GPS updates position/velocity; heading updates yaw; airspeed observes velocity norm under the still-air assumption. Sigma-point and noise parameters are in `configs/estimator.yaml`. Angular residuals wrap to [-π,π). Covariances are symmetrized and eigenvalue-floored for numerical stability; empirical consistency is evaluated separately from numerical PSD.

For each measurement update, `nu=z-z_pred`, `S=Cov(z_pred)+R`, and `NIS=nu.T solve(S,nu)`. Degrees of freedom are six for GPS and one for heading/airspeed. Full P, innovations, S, and NIS are retained in representative trajectory NPZs; scalar uncertainty summaries and NIS are available throughout training/evaluation. A sensor's NIS/dof uses EWMA with coefficient 0.1 per update and initial value one. The mean of the three sensor EWMAs forms the innovation-conditioning statistic. A missing measurement does not create a zero innovation: its EWMA is held while covariance can grow.

## Teacher and deployed student inputs

A 16→32→32→4 MLP, ReLU hidden layers clipped at six, is trained to imitate the conventional controller on complete trajectories. The policy outputs normalized aileron/elevator/rudder/throttle commands. Actuator clipping is common to all methods. The 16 memoryless features are normalized altitude/heading/airspeed tracking errors; estimated roll/pitch; bias-corrected three-axis gyro rates; estimated altitude and measured airspeed; sine/cosine of estimated heading; commanded altitude/airspeed; and sine/cosine of commanded heading. Exact ordering and fixed scales are in `policy_features(nav, ref)` in `control/autopilot.py`. This pure function accepts only physical navigation and reference records. Neural deployments do not construct or execute the conventional PID. The expert executes only when collecting demonstrations or evaluating the PID baseline. Checkpoints enforce the `physical_reference_v2` input schema and reject unversioned exploratory models.

Feature normalization is fitted on training trajectories only. FP32 biases, normalization, output affine scale, and actuator interface remain full precision. Teacher weights are frozen before student training. All students have the same architecture and deployment features. **P, NIS, truth, and fault labels are never student inputs.** This setup is not evidence that a neural controller is computationally preferable to the original PID.

## Quantization and optimization

Weights in every linear layer use per-tensor affine uniform fake quantization to `2^b` levels between the tensor's detached minimum and maximum. Hidden activations use uniform levels between zero and six. Two-bit weights therefore have four levels, not ternary weights. Feature input and final continuous output remain FP32; hidden activation and all linear-layer weight precision are b=8,4,2. A straight-through estimator is used. The current forward pass is floating-point fake quantization, not a hardware integer kernel. Approximate packed model storage includes weights, FP32 biases, normalization and quantizer parameters; no measured speed/energy/latency is claimed.

Teacher: 180 Adam epochs. Students: 100 Adam epochs, initialized from the identical frozen teacher for a given seed. Learning rate 0.001, batch size 1024. All QAD methods use matched minibatch order and the same training population, architecture and epoch budget. Checkpoints minimize unweighted action MSE on separate validation trajectories; evaluation rollouts never select checkpoints. Student loss is actuator-space MSE against frozen teacher outputs. PTQ uses identical quantization of the frozen teacher without optimization.

## Exact EC-QAD weighting

Select position and attitude covariance diagonals, indices I={0,1,2,6,7,8}. Define reference variance scales `a_i=25 m²` for position and `(5π/180)² rad²` for attitude. The dimensionless covariance statistic is

`c_t = log(1 + mean_i(P_ii(t)/a_i))`.

Let `n_t` be the mean sensor EWMA(NIS/dof), and `j_t=log(1+n_t)`. Standardize each statistic using training-population mean and standard deviation (floor 1e-6), then clip z-scores to [-3,3].

`q_t = 0.5 * (z_c(t)+z_j(t))/2` for combined EC-QAD;
`q_t = 0.5*z_c(t)` for covariance-only;
`q_t = 0.5*z_j(t)` for innovation-only.

Choose one scalar offset b by bisection on the training population so

`w_t = clip(exp(q_t+b), 0.5, 2.0)` and `mean_train(w_t)=1`.

The loss is `L = mean_batch[w_t * mean_action((u_T-u_S)^2)]`. Per-minibatch weight means fluctuate around one; no extra batch renormalization changes the bounds. This preserves mean training-population emphasis without accidentally scaling every gradient upward. Constants were fixed before evaluation. Shuffled weighting permutes the exact combined weights with a fixed seed, preserving their distribution. Fault-balanced weighting uses inverse counts of active training fault categories, passed through the same bound-and-mean-normalization procedure. It uses labels only during training.

## Interpretation limits

Correlations between estimator signals and student error do not establish causation. Near-future tracking error also contains nominal maneuver difficulty and estimator/controller errors. Conditional out-of-episode prediction and paired teacher comparisons are needed to assess whether information extends beyond fault category. Positive H1 need not imply positive H2. Synthetic sensors, approximate aircraft aerodynamics, a PID-imitation teacher, a narrow maneuver envelope, and fake quantization limit external validity.

The local truth position uses WGS84 meridional and prime-vertical radii to convert geodetic coordinates to a tangent-frame displacement. Earlier spherical-coordinate results are archived and excluded. The sensitivity ablation uses the Frobenius norm of the FP32 teacher action Jacobian with respect to standardized deployment features, followed by the same clipped standardized-log-score weighting and mean-one normalization. This is a simple sensitivity comparator, not an exact SQIL implementation.

Two label-based controls are distinguished: `balanced` is the original bounded inverse-frequency control and cannot exactly equalize rare-class mass because of clipping. `fault_balanced` uses `w=N/(K*n_class)` without clipping, giving each active-fault/nominal category exactly equal total training weight and population mean one. This stronger control retains the same sample count, epochs, optimizer and data; only its weighting differs. Its weights need not lie in EC-QAD's [0.5,2] range.
