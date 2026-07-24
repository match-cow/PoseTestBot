package application;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.nio.charset.Charset;

import javax.inject.Inject;

import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import com.kuka.roboticsAPI.deviceModel.JointPosition;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.geometricModel.ObjectFrame;
import com.kuka.roboticsAPI.motionModel.IMotionContainer;
import com.kuka.roboticsAPI.persistenceModel.templateModel.InfoTemplate;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

/**
 * Candidate ordinary full-capture application for the PoseTestBot iiwa.
 *
 * IMPORTANT: This source does not establish which application is deployed on
 * the lab controller. Compile and simulate it in the exact Sunrise.Workbench
 * project, teach and verify PoseTemplateBase, CaptureStart, and CaptureEnd,
 * and commission the complete PTP/A1/PTP path in T1 before setting
 * ENABLE_AFTER_OFFLINE_VALIDATION to true.
 *
 * The UDP command is read only while the application is idle. A UDP STOP
 * cannot interrupt an active motion and is not a safety stop.
 */
public class PoseTestBot_Test extends RoboticsAPIApplication {
	private static final String POSE_TEMPLATE_BASE_PATH =
			"/PoseTestBot/PoseTemplateBase";
	private static final String CAPTURE_START_FRAME_PATH =
			"/PoseTestBot/CaptureStart";
	private static final String CAPTURE_END_FRAME_PATH =
			"/PoseTestBot/CaptureEnd";
	private static final String POSE_PACKET_SCHEMA_VERSION = "robot_pose.v1";
	private static final String DEFAULT_RECEIVER_IP = "172.31.1.169";
	private static final Charset UTF_8 = Charset.forName("UTF-8");

	/*
	 * This repository source is intentionally inert until the exact controller
	 * project, frame, endpoints, joint branches, swept paths, tool/load, camera
	 * rig, and cables have been validated offline and commissioned in T1.
	 */
	private static final boolean ENABLE_AFTER_OFFLINE_VALIDATION = false;

	private static final int SAMPLE_TIME_MS = 10;
	private static final int SETTLE_TIME_MS = 1500;
	private static final int COMMAND_BUFFER_BYTES = 4096;
	private static final int ROBOT_PORT = 30300;
	private static final int DEFAULT_RECEIVER_PORT = 8080;
	private static final int END_PACKET_COUNT = 3;
	private static final int END_PACKET_INTERVAL_MS = 50;

	private static final double REPOSITION_JOINT_VEL_REL = 0.08;
	private static final double MAX_CAPTURE_CARTESIAN_VELOCITY_M_S = 0.03;
	private static final double MAX_CAPTURE_A1_ANGULAR_VELOCITY_RAD_S =
			Math.toRadians(3.0);
	private static final double SMOOTH_MOTION_JOINT_ACCEL_REL = 0.03;
	private static final double SMOOTH_MOTION_JOINT_JERK_REL = 0.03;
	private static final double MIN_A1_ORBIT_RADIUS_MM = 50.0;

	/*
	 * KUKA publishes 98 deg/s for A1 on the LBR iiwa 7 R800 and 85 deg/s on
	 * the LBR iiwa 14 R820. Using the larger value as the denominator makes
	 * the requested Cartesian speed an upper bound on either model. The exact
	 * installed model still has to be recorded during commissioning.
	 */
	private static final double A1_FULL_SPEED_UPPER_BOUND_RAD_S =
			Math.toRadians(98.0);
	private static final double A1_MIN_RAD = Math.toRadians(-169.0);
	private static final double A1_MAX_RAD = Math.toRadians(169.0);

	@Inject
	private LBR robot;
	@Inject
	private InfoTemplate robotinfo;

	private ObjectFrame poseTemplateBase;
	private ObjectFrame captureStartFrame;
	private ObjectFrame captureEndFrame;
	private JointPosition captureAnchorJointPosition;
	private long nextPacketSequence;

	@Override
	public void initialize() {
		robot = getContext().getDeviceFromType(LBR.class);
		robotinfo.setBase(POSE_TEMPLATE_BASE_PATH);
		poseTemplateBase = requiredFrame(POSE_TEMPLATE_BASE_PATH);
		captureStartFrame = requiredFrame(CAPTURE_START_FRAME_PATH);
		captureEndFrame = requiredFrame(CAPTURE_END_FRAME_PATH);

		getLogger().info("Resolved ordinary-capture reference frame: "
				+ POSE_TEMPLATE_BASE_PATH);
		getLogger().info("Resolved commissioned capture start frame: "
				+ CAPTURE_START_FRAME_PATH);
		getLogger().info("Resolved commissioned capture end frame: "
				+ CAPTURE_END_FRAME_PATH);
		getLogger().info("Configured application base: " + robotinfo.getBase());
		getLogger().info("Configured application tool: " + robotinfo.getTool());
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
			getLogger().error("Ordinary full-capture application is disabled. "
					+ "Complete Workbench and T1 commissioning before enabling it.");
			return;
		}

		getLogger().warn("UDP STOP is not a safety stop and cannot interrupt "
				+ "active motion. Use only the controller's approved safety "
				+ "response for an unsafe condition.");

		while (true) {
			CaptureCommand command = waitForStartCommand();
			if (command == null) {
				return;
			}

			nextPacketSequence = 0L;

			try {
				runCapture(command);
			} catch (RuntimeException e) {
				getLogger().error("Capture motion failed; no successful end "
						+ "marker will be reported: " + e);
				return;
			}
			if (Thread.currentThread().isInterrupted()) {
				getLogger().error("Capture thread was interrupted; exiting "
						+ "instead of accepting another start command");
				return;
			}
		}
	}

	private void runCapture(CaptureCommand command) {
		getLogger().info("Starting commissioned A1 sweep for run "
				+ command.runId + " with requested Cartesian speed "
				+ command.cartesianVelocityMps + " m/s");

		moveToCommissionedFrame(
				captureStartFrame,
				CAPTURE_START_FRAME_PATH,
				"capture start");
		/*
		 * Preserve the commissioned start frame's non-A1 joint branch during
		 * the single-axis sweep. No robot motion occurs before an accepted
		 * start command.
		 */
		captureAnchorJointPosition = robot.getCurrentJointPosition();
		moveToA1Min();
		sleepWithLogging(SETTLE_TIME_MS, "pre-capture settle");

		double captureJointVelocityRel = captureJointVelocityRel(
				command.cartesianVelocityMps);
		IMotionContainer motion = robot.moveAsync(ptp(jointTarget(A1_MAX_RAD))
				.setJointVelocityRel(captureJointVelocityRel)
				.setJointAccelerationRel(SMOOTH_MOTION_JOINT_ACCEL_REL)
				.setJointJerkRel(SMOOTH_MOTION_JOINT_JERK_REL));

		int sentPoseCount = transmitPose(motion, command, "a1_capture_sweep");
		getLogger().info("A1 sweep finished after transmitting "
				+ sentPoseCount + " pose packet(s)");
		sleepWithLogging(SETTLE_TIME_MS, "post-capture settle");
		moveToCommissionedFrame(
				captureEndFrame,
				CAPTURE_END_FRAME_PATH,
				"capture end");

		/*
		 * Report success only after the blocking end-frame PTP has completed,
		 * so the host cannot release the capture job while the robot is still
		 * executing the commissioned sequence.
		 */
		transmitEndMarker(command);
	}

	private void moveToCommissionedFrame(ObjectFrame targetFrame,
			String framePath, String label) {
		getLogger().info("Moving PTP to commissioned " + label
				+ " frame: " + framePath);
		robot.move(ptp(targetFrame)
				.setJointVelocityRel(REPOSITION_JOINT_VEL_REL)
				.setJointAccelerationRel(SMOOTH_MOTION_JOINT_ACCEL_REL)
				.setJointJerkRel(SMOOTH_MOTION_JOINT_JERK_REL));
	}

	private void moveToA1Min() {
		getLogger().info("Moving to commissioned A1 sweep start");
		robot.move(ptp(jointTarget(A1_MIN_RAD))
				.setJointVelocityRel(REPOSITION_JOINT_VEL_REL)
				.setJointAccelerationRel(SMOOTH_MOTION_JOINT_ACCEL_REL)
				.setJointJerkRel(SMOOTH_MOTION_JOINT_JERK_REL));
	}

	private JointPosition jointTarget(double a1Rad) {
		JointPosition target = new JointPosition(
				captureAnchorJointPosition.get(0),
				captureAnchorJointPosition.get(1),
				captureAnchorJointPosition.get(2),
				captureAnchorJointPosition.get(3),
				captureAnchorJointPosition.get(4),
				captureAnchorJointPosition.get(5),
				captureAnchorJointPosition.get(6));
		target.set(0, a1Rad);
		return target;
	}

	/**
	 * Convert the command's Cartesian metres/second into the relative A1 speed
	 * needed for the flange's circular orbit around the robot-root Z axis.
	 *
	 * This is deliberately an upper-bound conversion: the 98 deg/s denominator
	 * is the larger published A1 full speed across the iiwa 7 R800 and 14 R820.
	 */
	private double captureJointVelocityRel(double cartesianVelocityMps) {
		if (Double.isNaN(cartesianVelocityMps)
				|| Double.isInfinite(cartesianVelocityMps)
				|| cartesianVelocityMps <= 0.0) {
			throw new IllegalArgumentException(
					"cartesian_velocity_m_s must be finite and greater than zero");
		}

		Frame flangeInRobotRoot = robot.getCurrentCartesianPosition(
				robot.getFlange(), robot.getRootFrame());
		double orbitRadiusMm = Math.sqrt(
				flangeInRobotRoot.getX() * flangeInRobotRoot.getX()
				+ flangeInRobotRoot.getY() * flangeInRobotRoot.getY());
		if (Double.isNaN(orbitRadiusMm)
				|| Double.isInfinite(orbitRadiusMm)
				|| orbitRadiusMm < MIN_A1_ORBIT_RADIUS_MM) {
			throw new IllegalStateException("A1 flange-orbit radius is "
					+ orbitRadiusMm + " mm; require at least "
					+ MIN_A1_ORBIT_RADIUS_MM
					+ " mm for a stable Cartesian-speed conversion");
		}

		double boundedCartesianVelocityMps = Math.min(
				cartesianVelocityMps,
				MAX_CAPTURE_CARTESIAN_VELOCITY_M_S);
		if (boundedCartesianVelocityMps < cartesianVelocityMps) {
			getLogger().warn("Requested Cartesian capture speed "
					+ cartesianVelocityMps + " m/s exceeds the "
					+ MAX_CAPTURE_CARTESIAN_VELOCITY_M_S
					+ " m/s controller cap; applying the cap");
		}

		double requestedAngularVelocityRadS =
				boundedCartesianVelocityMps * 1000.0 / orbitRadiusMm;
		double appliedAngularVelocityRadS = Math.min(
				requestedAngularVelocityRadS,
				MAX_CAPTURE_A1_ANGULAR_VELOCITY_RAD_S);
		double appliedRelativeVelocity = appliedAngularVelocityRadS
				/ A1_FULL_SPEED_UPPER_BOUND_RAD_S;

		if (appliedAngularVelocityRadS
				< requestedAngularVelocityRadS) {
			getLogger().warn("Cartesian conversion needs A1 angular velocity "
					+ Math.toDegrees(requestedAngularVelocityRadS)
					+ " deg/s; the controller capture cap limits it to "
					+ Math.toDegrees(appliedAngularVelocityRadS)
					+ " deg/s");
		}
		getLogger().info("A1 Cartesian-speed conversion: radius="
				+ orbitRadiusMm + " mm, requested="
				+ cartesianVelocityMps + " m/s, bounded Cartesian="
				+ boundedCartesianVelocityMps + " m/s, applied angular="
				+ Math.toDegrees(appliedAngularVelocityRadS)
				+ " deg/s, applied joint velocity rel="
				+ appliedRelativeVelocity);
		return appliedRelativeVelocity;
	}

	private CaptureCommand waitForStartCommand() {
		DatagramSocket socket = null;
		try {
			socket = new DatagramSocket(ROBOT_PORT);
			getLogger().info("Waiting for UDP start command on port "
					+ ROBOT_PORT + "...");

			while (true) {
				byte[] receiveData = new byte[COMMAND_BUFFER_BYTES];
				DatagramPacket receivePacket = new DatagramPacket(
						receiveData, receiveData.length);
				try {
					socket.receive(receivePacket);
				} catch (IOException e) {
					getLogger().error("UDP command receive failed: " + e);
					return null;
				}
				try {
					String jsonMessage = new String(
							receivePacket.getData(),
							0,
							receivePacket.getLength(),
							UTF_8);
					JSONObject jsonObject = (JSONObject) new JSONParser().parse(
							jsonMessage);

					if (isStopCommand(jsonObject)) {
						getLogger().warn("UDP stop request received while idle. "
								+ "It cannot interrupt active motion, is not a "
								+ "safety stop, and will only exit this application.");
						return null;
					}

					CaptureCommand command = captureCommand(
							jsonObject, receivePacket);
					if (command != null) {
						getLogger().info("Accepted start command from "
								+ receivePacket.getAddress().getHostAddress()
								+ "; pose receiver target "
								+ command.receiverAddress.getHostAddress()
								+ ":" + command.receiverPort);
						return command;
					}

					getLogger().warn("Ignored UDP command without a supported "
							+ "start or stop request");
				} catch (Exception e) {
					getLogger().error("Rejected UDP command from "
							+ receivePacket.getAddress().getHostAddress()
							+ ": " + e);
				}
			}
		} catch (SocketException e) {
			getLogger().error("Unable to bind UDP command port "
					+ ROBOT_PORT + ": " + e);
			return null;
		} finally {
			if (socket != null) {
				socket.close();
			}
		}
	}

	private CaptureCommand captureCommand(JSONObject jsonObject,
			DatagramPacket receivePacket) throws IOException {
		Double velocity = startValue(jsonObject);
		if (velocity == null) {
			return null;
		}
		if (Double.isNaN(velocity.doubleValue())
				|| Double.isInfinite(velocity.doubleValue())
				|| velocity.doubleValue() <= 0.0) {
			throw new IllegalArgumentException(
					"cartesian_velocity_m_s must be finite and greater than zero");
		}

		String receiverIp = DEFAULT_RECEIVER_IP;
		Object receiverIpValue = jsonObject.get("receiver_ip");
		if (receiverIpValue != null) {
			String requestedReceiverIp = receiverIpValue.toString().trim();
			if (requestedReceiverIp.length() == 0
					|| requestedReceiverIp.equals("0.0.0.0")
					|| requestedReceiverIp.equals("::")) {
				receiverIp = receivePacket.getAddress().getHostAddress();
			} else {
				receiverIp = requestedReceiverIp;
			}
		}

		int receiverPort = DEFAULT_RECEIVER_PORT;
		Object receiverPortValue = jsonObject.get("receiver_port");
		if (receiverPortValue != null) {
			receiverPort = integerValue(
					receiverPortValue, "receiver_port");
		}
		if (receiverPort < 1 || receiverPort > 65535) {
			throw new IllegalArgumentException(
					"receiver_port must be between 1 and 65535");
		}

		String runId = "legacy-" + System.currentTimeMillis();
		Object runIdValue = jsonObject.get("run_id");
		if (runIdValue != null && runIdValue.toString().trim().length() > 0) {
			runId = runIdValue.toString().trim();
		}

		return new CaptureCommand(
				velocity.doubleValue(),
				InetAddress.getByName(receiverIp),
				receiverPort,
				runId);
	}

	private Double startValue(JSONObject jsonObject) {
		Object legacyStartValue = jsonObject.get("start");
		if (legacyStartValue != null) {
			return doubleValue(legacyStartValue);
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

	private int integerValue(Object value, String name) {
		if (value instanceof Number) {
			double number = ((Number) value).doubleValue();
			if (Double.isNaN(number)
					|| Double.isInfinite(number)
					|| number != Math.rint(number)
					|| number < Integer.MIN_VALUE
					|| number > Integer.MAX_VALUE) {
				throw new IllegalArgumentException(
						name + " must be an integer");
			}
			return (int) number;
		}
		return Integer.parseInt(value.toString());
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

	@SuppressWarnings("unchecked")
	private byte[] currentPosePayload(String motionName,
			CaptureCommand command) {
		Frame currentPose = robot.getCurrentCartesianPosition(
				robot.getFlange(), poseTemplateBase);
		JSONObject jsonObject = packetEnvelope(
				"pose", motionName, command, nextPacketSequence++);
		jsonObject.put("X", currentPose.getX());
		jsonObject.put("Y", currentPose.getY());
		jsonObject.put("Z", currentPose.getZ());
		jsonObject.put("A", currentPose.getAlphaRad());
		jsonObject.put("B", currentPose.getBetaRad());
		jsonObject.put("C", currentPose.getGammaRad());
		return jsonObject.toJSONString().getBytes(UTF_8);
	}

	@SuppressWarnings("unchecked")
	private JSONObject packetEnvelope(String packetKind, String motionName,
			CaptureCommand command, long sequence) {
		JSONObject jsonObject = new JSONObject();
		jsonObject.put("schema_version", POSE_PACKET_SCHEMA_VERSION);
		jsonObject.put("packet_kind", packetKind);
		jsonObject.put("sequence", Long.valueOf(sequence));
		jsonObject.put("sender_monotonic_ns",
				Long.valueOf(System.nanoTime()));
		jsonObject.put("sender_wall_timestamp_ms",
				Long.valueOf(System.currentTimeMillis()));
		jsonObject.put("run_id", command.runId);
		jsonObject.put("motion", motionName);
		jsonObject.put("from_frame", "robot_flange");
		jsonObject.put("to_frame", "template_base");
		jsonObject.put("sunrise_reference_frame_path",
				POSE_TEMPLATE_BASE_PATH);
		return jsonObject;
	}

	private void sendCurrentPose(DatagramSocket socket,
			CaptureCommand command, String motionName) throws IOException {
		byte[] data = currentPosePayload(motionName, command);
		sendPayload(socket, command, data);
	}

	private void sendPayload(DatagramSocket socket, CaptureCommand command,
			byte[] data) throws IOException {
		DatagramPacket packet = new DatagramPacket(
				data,
				data.length,
				command.receiverAddress,
				command.receiverPort);
		socket.send(packet);
	}

	private int transmitPose(IMotionContainer motion, CaptureCommand command,
			String motionName) {
		DatagramSocket socket = null;
		int sentPoseCount = 0;
		int sendFailureCount = 0;
		boolean interrupted = false;

		try {
			socket = new DatagramSocket();
			while (!motion.isFinished()) {
				try {
					sendCurrentPose(socket, command, motionName);
					sentPoseCount++;
				} catch (IOException e) {
					sendFailureCount++;
					if (sendFailureCount == 1
							|| sendFailureCount % 100 == 0) {
						getLogger().error("Pose packet transmission failure "
								+ sendFailureCount + ": " + e);
					}
				}

				try {
					Thread.sleep(SAMPLE_TIME_MS);
				} catch (InterruptedException e) {
					/*
					 * Do not report capture completion while the asynchronous
					 * robot motion is still active. Remember the interruption,
					 * continue observing the motion, and restore it afterward.
					 */
					interrupted = true;
					getLogger().error("Pose transmission wait interrupted; "
							+ "continuing until the active motion finishes");
				}
			}
		} catch (SocketException e) {
			getLogger().error("Unable to open pose-transmission socket: " + e);
			interrupted = waitForMotionCompletion(motion) || interrupted;
		} catch (RuntimeException e) {
			getLogger().error("Unable to build a pose packet; waiting for the "
					+ "active motion to finish before failing capture: " + e);
			interrupted = waitForMotionCompletion(motion) || interrupted;
			throw e;
		} finally {
			if (socket != null) {
				socket.close();
			}
			if (sendFailureCount > 0) {
				getLogger().warn("Pose stream completed with "
						+ sendFailureCount + " observable UDP send failure(s)");
			}
			if (interrupted) {
				Thread.currentThread().interrupt();
			}
		}
		return sentPoseCount;
	}

	@SuppressWarnings("unchecked")
	private void transmitEndMarker(CaptureCommand command) {
		DatagramSocket socket = null;
		long endSequence = nextPacketSequence++;
		JSONObject endObject = packetEnvelope(
				"end", "end", command, endSequence);
		byte[] data = endObject.toJSONString().getBytes(UTF_8);

		try {
			socket = new DatagramSocket();
			for (int i = 0; i < END_PACKET_COUNT; i++) {
				try {
					sendPayload(socket, command, data);
				} catch (IOException e) {
					getLogger().error("End marker transmission "
							+ (i + 1) + " failed: " + e);
				}
				if (i + 1 < END_PACKET_COUNT) {
					sleepWithLogging(
							END_PACKET_INTERVAL_MS,
							"end-marker retry interval");
				}
			}
		} catch (SocketException e) {
			getLogger().error("Unable to open end-marker socket: " + e);
		} finally {
			if (socket != null) {
				socket.close();
			}
		}
	}

	private boolean waitForMotionCompletion(IMotionContainer motion) {
		boolean interrupted = false;
		while (!motion.isFinished()) {
			try {
				Thread.sleep(SAMPLE_TIME_MS);
			} catch (InterruptedException e) {
				interrupted = true;
			}
		}
		return interrupted;
	}

	private void sleepWithLogging(int millis, String reason) {
		try {
			Thread.sleep(millis);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			getLogger().error("Interrupted during " + reason + ": " + e);
		}
	}

	private static final class CaptureCommand {
		private final double cartesianVelocityMps;
		private final InetAddress receiverAddress;
		private final int receiverPort;
		private final String runId;

		private CaptureCommand(double cartesianVelocityMps,
				InetAddress receiverAddress, int receiverPort, String runId) {
			this.cartesianVelocityMps = cartesianVelocityMps;
			this.receiverAddress = receiverAddress;
			this.receiverPort = receiverPort;
			this.runId = runId;
		}
	}
}
