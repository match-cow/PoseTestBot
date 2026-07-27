package application;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.util.concurrent.TimeUnit;

import javax.inject.Inject;

import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.geometricModel.ObjectFrame;
import com.kuka.roboticsAPI.geometricModel.math.Transformation;
import com.kuka.roboticsAPI.motionModel.IMotionContainer;
import com.kuka.roboticsAPI.persistenceModel.templateModel.InfoTemplate;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

/**
 * Proposal for an ArUco calibration capture with more image-space and
 * orientation variance than PoseTestBot_Test's single-axis A1 sweep.
 *
 * IMPORTANT: This repository revision is enabled for lab validation, but the
 * exact deployed controller application and revision must be independently
 * recorded. Revalidate every frame and connecting motion in Sunrise.Workbench
 * and T1 whenever the frames, tool, cables, target, cameras, or safety setup
 * change.
 *
 * The nine raster targets are persistent Application Data ObjectFrames directly
 * below /PoseTestBot/TemplateBase. Numeric values in the repository teaching
 * manifest are uncommissioned Workbench seeds only. The A/B/C orientation
 * dither is implemented as program-owned relative rotations from the taught
 * CalibrationCenter; no numeric absolute target is created at runtime.
 */
public class PoseTestBot_CalibrationVarianceProposal
		extends RoboticsAPIApplication {

	private static final String TEMPLATE_BASE_PATH = "/PoseTestBot/TemplateBase";
	private static final String CALIBRATION_COVERAGE_UPPER_LEFT_PATH = "/PoseTestBot/TemplateBase/CalibrationCoverageUpperLeft";
	private static final String CALIBRATION_COVERAGE_UPPER_CENTER_PATH = "/PoseTestBot/TemplateBase/CalibrationCoverageUpperCenter";
	private static final String CALIBRATION_COVERAGE_UPPER_RIGHT_PATH = "/PoseTestBot/TemplateBase/CalibrationCoverageUpperRight";
	private static final String CALIBRATION_COVERAGE_MIDDLE_RIGHT_PATH = "/PoseTestBot/TemplateBase/CalibrationCoverageMiddleRight";
	private static final String CALIBRATION_CENTER_PATH = "/PoseTestBot/TemplateBase/CalibrationCenter";
	private static final String CALIBRATION_COVERAGE_MIDDLE_LEFT_PATH = "/PoseTestBot/TemplateBase/CalibrationCoverageMiddleLeft";
	private static final String CALIBRATION_COVERAGE_LOWER_LEFT_PATH = "/PoseTestBot/TemplateBase/CalibrationCoverageLowerLeft";
	private static final String CALIBRATION_COVERAGE_LOWER_CENTER_PATH = "/PoseTestBot/TemplateBase/CalibrationCoverageLowerCenter";
	private static final String CALIBRATION_COVERAGE_LOWER_RIGHT_PATH = "/PoseTestBot/TemplateBase/CalibrationCoverageLowerRight";

	/* Disabled until offline review and supervised Workbench commissioning pass. */
	private static final boolean ENABLE_AFTER_OFFLINE_VALIDATION = false;
	/* Commission one phase at a time before enabling both together. */
	private static final boolean RUN_COVERAGE_RASTER = true;
	private static final boolean RUN_ORIENTATION_DITHER = true;

	private static final int SAMPLE_TIME_MS = 10;
	private static final int SETTLE_TIME_MS = 1500;
	private static final int ROBOT_PORT = 30300;
	private static final int DEFAULT_RECEIVER_PORT = 8080;
	private static final int END_PACKET_COUNT = 3;
	private static final int END_PACKET_INTERVAL_MS = 50;

	/* Keep capture motion below the requested run velocity to limit blur. */
	private static final double CAPTURE_VELOCITY_SCALE = 0.60;
	private static final double REPOSITION_PTP_VEL_REL = 0.08;
	private static final double ORIENTATION_JOINT_VEL_REL = 0.03;
	private static final double SMOOTH_MOTION_JOINT_ACCEL_REL = 0.03;
	private static final double SMOOTH_MOTION_JOINT_JERK_REL = 0.03;
	private static final double MIN_CART_VEL_MM_S = 8.0;
	private static final double MAX_CART_VEL_MM_S = 30.0;

	@Inject
	private LBR robot;
	@Inject
	private InfoTemplate robotinfo;

	private ObjectFrame templateBase;
	private ObjectFrame coverageUpperLeft;
	private ObjectFrame coverageUpperCenter;
	private ObjectFrame coverageUpperRight;
	private ObjectFrame coverageMiddleRight;
	private ObjectFrame calibrationCenter;
	private ObjectFrame coverageMiddleLeft;
	private ObjectFrame coverageLowerLeft;
	private ObjectFrame coverageLowerCenter;
	private ObjectFrame coverageLowerRight;
	private String receiverIp = "172.31.1.169";
	private int receiverPort = DEFAULT_RECEIVER_PORT;

	@Override
	public void initialize() {
		robot = getContext().getDeviceFromType(LBR.class);
		robotinfo.setBase(TEMPLATE_BASE_PATH);
		templateBase = requiredFrame(TEMPLATE_BASE_PATH);
		coverageUpperLeft = requiredFrame(CALIBRATION_COVERAGE_UPPER_LEFT_PATH);
		coverageUpperCenter = requiredFrame(CALIBRATION_COVERAGE_UPPER_CENTER_PATH);
		coverageUpperRight = requiredFrame(CALIBRATION_COVERAGE_UPPER_RIGHT_PATH);
		coverageMiddleRight = requiredFrame(CALIBRATION_COVERAGE_MIDDLE_RIGHT_PATH);
		calibrationCenter = requiredFrame(CALIBRATION_CENTER_PATH);
		coverageMiddleLeft = requiredFrame(CALIBRATION_COVERAGE_MIDDLE_LEFT_PATH);
		coverageLowerLeft = requiredFrame(CALIBRATION_COVERAGE_LOWER_LEFT_PATH);
		coverageLowerCenter = requiredFrame(CALIBRATION_COVERAGE_LOWER_CENTER_PATH);
		coverageLowerRight = requiredFrame(CALIBRATION_COVERAGE_LOWER_RIGHT_PATH);

		getLogger().info("Resolved TemplateBase and all nine taught grid frames: "
				+ robotinfo.getBase());
	}

	private ObjectFrame requiredFrame(String path) {
		ObjectFrame frame = getApplicationData().getFrame(path);
		if (frame == null) {
			throw new IllegalStateException(
					"Required Application Data frame is missing: " + path);
		}
		return frame;
	}

	@Override
	public void run() {
		if (!ENABLE_AFTER_OFFLINE_VALIDATION) {
			getLogger().error("Calibration variance proposal is disabled. "
					+ "Commission all frames and motions before enabling it.");
			return;
		}

		getLogger().warn("Before the first start command, manually position the "
				+ "robot at or near the taught CalibrationCenter pose. This is an "
				+ "operator commissioning requirement, not an enforced safety check.");

		while (true) {
			Double requestedVelocityMps = waitForStartCommand();
			if (requestedVelocityMps == null) {
				return;
			}

			double cartVelocityMmS = cartVelocityMmS(
					requestedVelocityMps.doubleValue());
			getLogger().info("Starting calibration variance capture at "
					+ cartVelocityMmS + " mm/s");
			moveToCenter("capture start anchor");

			if (RUN_COVERAGE_RASTER) {
				runCoverageRaster(cartVelocityMmS);
			}
			if (RUN_ORIENTATION_DITHER) {
				runOrientationDither(cartVelocityMmS);
			}

			transmitCurrentPose("end");
			sleep(SETTLE_TIME_MS);
		}
	}

	/**
	 * Raster translation is the primary image-centroid coverage mechanism.
	 * The +/-160 mm lateral offsets and roughly +/-90 mm vertical offsets are
	 * intended to cross the thirds of a 1280x720 image without re-aiming the
	 * camera perfectly at the board on every waypoint.
	 */
	private void runCoverageRaster(double cartVelocityMmS) {
		moveFromCenter(coverageUpperLeft, "coverage raster");
		captureLinear(coverageUpperCenter, cartVelocityMmS,
				"coverage_upper_left_to_center");
		captureLinear(coverageUpperRight, cartVelocityMmS,
				"coverage_upper_center_to_right");
		captureLinear(coverageMiddleRight, cartVelocityMmS,
				"coverage_upper_to_middle_right");
		captureLinear(calibrationCenter, cartVelocityMmS,
				"coverage_middle_right_to_center");
		captureLinear(coverageMiddleLeft, cartVelocityMmS,
				"coverage_middle_center_to_left");
		captureLinear(coverageLowerLeft, cartVelocityMmS,
				"coverage_middle_to_lower_left");
		captureLinear(coverageLowerCenter, cartVelocityMmS,
				"coverage_lower_left_to_center");
		captureLinear(coverageLowerRight, cartVelocityMmS,
				"coverage_lower_center_to_right");
		moveToCenter("coverage raster return");
	}

	/**
	 * Adds rotation-axis diversity for intrinsic and hand-eye observability.
	 * These are intentionally modest +/-15 degree A/C and +/-12 degree B
	 * offsets. Which change maps to image yaw, pitch, or roll depends on the
	 * actual flange-to-camera mounting transform and must be verified visually.
	 */
	private void runOrientationDither(double cartVelocityMmS) {
		captureRelativeOrientation(-15, 0, 0, cartVelocityMmS,
				"orientation_alpha_minus_15");
		captureRelativeOrientation(30, 0, 0, cartVelocityMmS,
				"orientation_alpha_plus_15");
		captureRelativeOrientation(-15, 0, 0, cartVelocityMmS,
				"orientation_alpha_return_center");
		captureRelativeOrientation(0, -12, 0, cartVelocityMmS,
				"orientation_beta_minus_12");
		captureRelativeOrientation(0, 24, 0, cartVelocityMmS,
				"orientation_beta_plus_12");
		captureRelativeOrientation(0, -12, 0, cartVelocityMmS,
				"orientation_beta_return_center");
		captureRelativeOrientation(0, 0, -15, cartVelocityMmS,
				"orientation_gamma_minus_15");
		captureRelativeOrientation(0, 0, 30, cartVelocityMmS,
				"orientation_gamma_plus_15");
		captureRelativeOrientation(0, 0, -15, cartVelocityMmS,
				"orientation_gamma_return_center");
	}

	private void moveFromCenter(ObjectFrame target, String phaseName) {
		getLogger().info("PTP from taught CalibrationCenter into " + phaseName);
		robot.move(ptp(target)
				.setJointVelocityRel(REPOSITION_PTP_VEL_REL)
				.setJointAccelerationRel(SMOOTH_MOTION_JOINT_ACCEL_REL)
				.setJointJerkRel(SMOOTH_MOTION_JOINT_JERK_REL));
		settleAtCurrentPose(phaseName);
	}

	private void moveToCenter(String motionName) {
		getLogger().info("PTP to taught CalibrationCenter: " + motionName);
		robot.move(ptp(calibrationCenter)
				.setJointVelocityRel(REPOSITION_PTP_VEL_REL)
				.setJointAccelerationRel(SMOOTH_MOTION_JOINT_ACCEL_REL)
				.setJointJerkRel(SMOOTH_MOTION_JOINT_JERK_REL));
		settleAtCurrentPose(motionName);
	}

	private void captureLinear(ObjectFrame target, double cartVelocityMmS,
			String motionName) {
		IMotionContainer motion = robot.moveAsync(lin(target)
				.setCartVelocity(cartVelocityMmS)
				.setJointAccelerationRel(SMOOTH_MOTION_JOINT_ACCEL_REL)
				.setJointJerkRel(SMOOTH_MOTION_JOINT_JERK_REL));
		transmitPose(motion, SAMPLE_TIME_MS, motionName);
		settleAtCurrentPose(motionName);
	}

	private void captureRelativeOrientation(double alphaDeg, double betaDeg,
			double gammaDeg, double cartVelocityMmS, String motionName) {
		Transformation offset = Transformation.ofDeg(0, 0, 0,
				alphaDeg, betaDeg, gammaDeg);
		IMotionContainer motion = robot.moveAsync(linRel(offset,
				calibrationCenter).setCartVelocity(cartVelocityMmS)
				.setJointVelocityRel(ORIENTATION_JOINT_VEL_REL)
				.setJointAccelerationRel(SMOOTH_MOTION_JOINT_ACCEL_REL)
				.setJointJerkRel(SMOOTH_MOTION_JOINT_JERK_REL));
		transmitPose(motion, SAMPLE_TIME_MS, motionName);
		settleAtCurrentPose(motionName);
	}

	private void settleAtCurrentPose(String motionName) {
		sleep(SETTLE_TIME_MS);
		transmitCurrentPose(motionName + "_settled");
	}

	private double cartVelocityMmS(double requestedMps) {
		if (Double.isNaN(requestedMps) || Double.isInfinite(requestedMps)
				|| requestedMps <= 0.0) {
			throw new IllegalArgumentException(
					"Capture velocity must be a finite positive value in m/s");
		}

		double requestedMmS = requestedMps * 1000.0;
		double scaledMmS = requestedMmS * CAPTURE_VELOCITY_SCALE;
		double clampedMmS = Math.max(MIN_CART_VEL_MM_S,
				Math.min(MAX_CART_VEL_MM_S, scaledMmS));
		if (clampedMmS != scaledMmS) {
			getLogger().warn("Requested " + requestedMmS
					+ " mm/s; the calibration speed scale gives "
					+ scaledMmS + " mm/s and the configured bounds clamp "
					+ "this to " + clampedMmS + " mm/s");
		} else {
			getLogger().info("Requested " + requestedMmS
					+ " mm/s; applying calibration speed scale: "
					+ clampedMmS + " mm/s");
		}
		return clampedMmS;
	}

	private Double waitForStartCommand() {
		while (true) {
			DatagramSocket socket = null;

			try {
				socket = new DatagramSocket(ROBOT_PORT);
				getLogger().info("Waiting for UDP start command...");

				byte[] receiveData = new byte[1024];
				DatagramPacket receivePacket = new DatagramPacket(
						receiveData, receiveData.length);
				socket.receive(receivePacket);

				String jsonMessage = new String(receivePacket.getData(), 0,
						receivePacket.getLength());
				JSONObject jsonObject = (JSONObject) new JSONParser().parse(
						jsonMessage);

				Double startValue = startValue(jsonObject);
				if (startValue != null) {
					updateReceiverTarget(jsonObject, receivePacket);
					return startValue;
				}

				if (isStopCommand(jsonObject)) {
					getLogger().info("Stop message received. Ending program.");
					return null;
				}
			} catch (Exception e) {
				getLogger().error("UDP command error: " + e);
			} finally {
				if (socket != null) {
					socket.close();
				}
			}
		}
	}

	private Double startValue(JSONObject jsonObject) {
		Object startValue = jsonObject.get("start");
		if (startValue != null) {
			return doubleValue(startValue);
		}

		if ("start_capture".equals(jsonObject.get("command"))) {
			return doubleValue(jsonObject.get("cartesian_velocity_m_s"));
		}

		return null;
	}

	private Double doubleValue(Object value) {
		if (value == null) {
			return null;
		}
		if (value instanceof Number) {
			return Double.valueOf(((Number) value).doubleValue());
		}
		return Double.valueOf(value.toString());
	}

	private boolean isStopCommand(JSONObject jsonObject) {
		if (Boolean.TRUE.equals(jsonObject.get("stop"))) {
			return true;
		}

		Object command = jsonObject.get("command");
		return "pause_capture".equals(command)
				|| "stop_after_current_motion".equals(command)
				|| "emergency_stop".equals(command);
	}

	private void updateReceiverTarget(JSONObject jsonObject,
			DatagramPacket receivePacket) {
		Object receiverIpValue = jsonObject.get("receiver_ip");
		if (receiverIpValue != null) {
			String requestedReceiverIp = receiverIpValue.toString().trim();
			if (requestedReceiverIp.length() > 0
					&& !requestedReceiverIp.equals("0.0.0.0")
					&& !requestedReceiverIp.equals("::")) {
				receiverIp = requestedReceiverIp;
			} else {
				receiverIp = receivePacket.getAddress().getHostAddress();
			}
		} else {
			receiverIp = receivePacket.getAddress().getHostAddress();
		}

		Object receiverPortValue = jsonObject.get("receiver_port");
		if (receiverPortValue instanceof Number) {
			receiverPort = ((Number) receiverPortValue).intValue();
		} else if (receiverPortValue != null) {
			receiverPort = Integer.parseInt(receiverPortValue.toString());
		}

		getLogger().info("Pose receiver target: " + receiverIp + ":"
				+ receiverPort);
	}

	@SuppressWarnings("unchecked")
	private byte[] currentPosePayload(String motionName) {
		Frame currentPose = robot.getCurrentCartesianPosition(robot.getFlange(),
				templateBase);

		JSONObject jsonObject = new JSONObject();
		jsonObject.put("motion", motionName);
		jsonObject.put("X", currentPose.getX());
		jsonObject.put("Y", currentPose.getY());
		jsonObject.put("Z", currentPose.getZ());
		jsonObject.put("A", currentPose.getAlphaRad());
		jsonObject.put("B", currentPose.getBetaRad());
		jsonObject.put("C", currentPose.getGammaRad());

		return jsonObject.toJSONString().getBytes();
	}

	private void sendCurrentPose(DatagramSocket socket, String motionName)
			throws IOException {
		byte[] data = currentPosePayload(motionName);
		DatagramPacket packet = new DatagramPacket(data, data.length,
				InetAddress.getByName(receiverIp), receiverPort);
		socket.send(packet);
	}

	private void transmitCurrentPose(String motionName) {
		DatagramSocket socket = null;

		try {
			socket = new DatagramSocket();
			for (int i = 0; i < END_PACKET_COUNT; i++) {
				sendCurrentPose(socket, motionName);
				TimeUnit.MILLISECONDS.sleep(END_PACKET_INTERVAL_MS);
			}
		} catch (Exception e) {
			getLogger().error("Unable to transmit current pose: " + e);
		} finally {
			if (socket != null) {
				socket.close();
			}
		}
	}

	private void transmitPose(IMotionContainer motion, int sampleTimeMs,
			String motionName) {
		DatagramSocket socket = null;

		try {
			socket = new DatagramSocket();
			while (!motion.isFinished()) {
				sendCurrentPose(socket, motionName);
				TimeUnit.MILLISECONDS.sleep(sampleTimeMs);
			}
		} catch (SocketException e) {
			getLogger().error("Unable to open pose socket: " + e);
		} catch (IOException e) {
			getLogger().error("Unable to transmit pose: " + e);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			getLogger().error("Pose transmission interrupted: " + e);
		} finally {
			if (socket != null) {
				socket.close();
			}
		}
	}

	private void sleep(int millis) {
		try {
			Thread.sleep(millis);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}
}
