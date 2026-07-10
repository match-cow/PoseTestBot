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
import com.kuka.roboticsAPI.geometricModel.AbstractFrame;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.geometricModel.ObjectFrame;
import com.kuka.roboticsAPI.motionModel.IMotionContainer;
import com.kuka.roboticsAPI.persistenceModel.templateModel.InfoTemplate;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

/**
 * Proposal for an ArUco calibration capture with more image-space and
 * orientation variance than PoseTestBot_Test's single-axis A1 sweep.
 *
 * IMPORTANT: The proposed Cartesian frames have not been commissioned on the
 * physical cell. Keep ENABLE_AFTER_OFFLINE_VALIDATION false until every frame
 * and connecting motion has been checked in Sunrise.Workbench and then
 * single-stepped in T1 at reduced override with the real tool, cables, target,
 * cameras, and safety equipment represented.
 *
 * Coordinates are relative to /HRC_Hub/Template_Base. They stay inside the
 * envelope suggested by HRC_Hub_Cap, but that is not proof of reachability,
 * collision freedom, singularity freedom, or target visibility.
 */
public class PoseTestBot_CalibrationVarianceProposal
		extends RoboticsAPIApplication {

	/* Deliberate deployment interlock for this uncommissioned proposal. */
	private static final boolean ENABLE_AFTER_OFFLINE_VALIDATION = false;
	/* Commission one phase at a time before enabling all three together. */
	private static final boolean RUN_COVERAGE_RASTER = true;
	private static final boolean RUN_DEPTH_SWEEP = true;
	private static final boolean RUN_ORIENTATION_DITHER = true;

	private static final int SAMPLE_TIME_MS = 10;
	private static final int SETTLE_TIME_MS = 500;
	private static final int ROBOT_PORT = 30300;
	private static final int DEFAULT_RECEIVER_PORT = 8080;
	private static final int END_PACKET_COUNT = 3;
	private static final int END_PACKET_INTERVAL_MS = 50;

	private static final double REPOSITION_PTP_VEL_REL = 0.15;
	private static final double CAPTURE_PTP_VEL_REL = 0.10;
	private static final double MIN_CART_VEL_MM_S = 20.0;
	private static final double MAX_CART_VEL_MM_S = 80.0;

	@Inject
	private LBR robot;
	@Inject
	private InfoTemplate robotinfo;

	private AbstractFrame templateBaseRef;
	private ObjectFrame templateBase;
	private String receiverIp = "172.31.1.169";
	private int receiverPort = DEFAULT_RECEIVER_PORT;

	@Override
	public void initialize() {
		robot = getContext().getDeviceFromType(LBR.class);
		robotinfo.setBase("/HRC_Hub/Template_Base");
		templateBaseRef = getApplicationData().getFrame(
				"/HRC_Hub/Template_Base");
		templateBase = getApplicationData().getFrame(
				"/HRC_Hub/Template_Base");

		getLogger().info("Calibration variance proposal base: "
				+ robotinfo.getBase());
	}

	@Override
	public void run() {
		if (!ENABLE_AFTER_OFFLINE_VALIDATION) {
			getLogger().error("Calibration variance proposal is disabled. "
					+ "Commission all frames and motions before enabling it.");
			return;
		}

		while (true) {
			Double requestedVelocityMps = waitForStartCommand();
			if (requestedVelocityMps == null) {
				return;
			}

			double cartVelocityMmS = cartVelocityMmS(
					requestedVelocityMps.doubleValue());
			getLogger().info("Starting calibration variance capture at "
					+ cartVelocityMmS + " mm/s");

			if (RUN_COVERAGE_RASTER) {
				runCoverageRaster(cartVelocityMmS);
			}
			if (RUN_DEPTH_SWEEP) {
				runDepthSweep(cartVelocityMmS);
			}
			if (RUN_ORIENTATION_DITHER) {
				runOrientationDither();
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
		Frame upperLeft = frame(-160, -320, 535, -90, 24, 180);
		Frame upperCenter = frame(0, -320, 535, -90, 24, 180);
		Frame upperRight = frame(160, -320, 535, -90, 24, 180);

		Frame middleRight = frame(160, -285, 445, -90, 30, 180);
		Frame middleCenter = frame(0, -285, 445, -90, 30, 180);
		Frame middleLeft = frame(-160, -285, 445, -90, 30, 180);

		Frame lowerLeft = frame(-160, -245, 355, -90, 36, 180);
		Frame lowerCenter = frame(0, -245, 355, -90, 36, 180);
		Frame lowerRight = frame(160, -245, 355, -90, 36, 180);

		moveToStart(upperLeft, "coverage raster");
		captureLinear(upperCenter, cartVelocityMmS,
				"coverage_upper_left_to_center");
		captureLinear(upperRight, cartVelocityMmS,
				"coverage_upper_center_to_right");
		captureLinear(middleRight, cartVelocityMmS,
				"coverage_upper_to_middle_right");
		captureLinear(middleCenter, cartVelocityMmS,
				"coverage_middle_right_to_center");
		captureLinear(middleLeft, cartVelocityMmS,
				"coverage_middle_center_to_left");
		captureLinear(lowerLeft, cartVelocityMmS,
				"coverage_middle_to_lower_left");
		captureLinear(lowerCenter, cartVelocityMmS,
				"coverage_lower_left_to_center");
		captureLinear(lowerRight, cartVelocityMmS,
				"coverage_lower_center_to_right");
	}

	/**
	 * Varies board scale and perspective. The endpoints are based on the
	 * Center/CenterClose/Top/Bottom envelope in HRC_Hub_Cap, not on a new
	 * reachability study.
	 */
	private void runDepthSweep(double cartVelocityMmS) {
		Frame farCenter = frame(0, -360, 600, -90, 20, 180);
		Frame nearCenter = frame(0, -230, 350, -90, 38, 180);

		moveToStart(farCenter, "depth sweep");
		captureLinear(nearCenter, cartVelocityMmS,
				"depth_far_to_near");
		captureLinear(farCenter, cartVelocityMmS,
				"depth_near_to_far");
	}

	/**
	 * Adds rotation-axis diversity for intrinsic and hand-eye observability.
	 * These are intentionally modest +/-15 degree A/C and +/-12 degree B
	 * offsets. Which change maps to image yaw, pitch, or roll depends on the
	 * actual flange-to-camera mounting transform and must be verified visually.
	 */
	private void runOrientationDither() {
		Frame center = frame(0, -285, 445, -90, 30, 180);
		Frame alphaMinus = frame(0, -285, 445, -105, 30, 180);
		Frame alphaPlus = frame(0, -285, 445, -75, 30, 180);
		Frame betaMinus = frame(0, -285, 445, -90, 18, 180);
		Frame betaPlus = frame(0, -285, 445, -90, 42, 180);
		Frame gammaMinus = frame(0, -285, 445, -90, 30, 165);
		Frame gammaPlus = frame(0, -285, 445, -90, 30, -165);

		moveToStart(center, "orientation dither");
		capturePointToPoint(alphaMinus, "orientation_alpha_minus_15");
		capturePointToPoint(alphaPlus, "orientation_alpha_plus_15");
		capturePointToPoint(center, "orientation_alpha_return_center");
		capturePointToPoint(betaMinus, "orientation_beta_minus_12");
		capturePointToPoint(betaPlus, "orientation_beta_plus_12");
		capturePointToPoint(center, "orientation_beta_return_center");
		capturePointToPoint(gammaMinus, "orientation_gamma_minus_15");
		capturePointToPoint(gammaPlus, "orientation_gamma_plus_15");
		capturePointToPoint(center, "orientation_gamma_return_center");
	}

	private Frame frame(double xMm, double yMm, double zMm,
			double alphaDeg, double betaDeg, double gammaDeg) {
		Frame target = new Frame(xMm, yMm, zMm,
				Math.toRadians(alphaDeg), Math.toRadians(betaDeg),
				Math.toRadians(gammaDeg));
		target.setParent(templateBase);
		return target;
	}

	private void moveToStart(Frame target, String phaseName) {
		getLogger().info("Repositioning for " + phaseName);
		robot.move(ptp(target).setJointVelocityRel(REPOSITION_PTP_VEL_REL));
		sleep(SETTLE_TIME_MS);
	}

	private void captureLinear(Frame target, double cartVelocityMmS,
			String motionName) {
		IMotionContainer motion = robot.moveAsync(lin(target)
				.setCartVelocity(cartVelocityMmS));
		transmitPose(motion, SAMPLE_TIME_MS, motionName);
	}

	private void capturePointToPoint(Frame target, String motionName) {
		IMotionContainer motion = robot.moveAsync(ptp(target)
				.setJointVelocityRel(CAPTURE_PTP_VEL_REL));
		transmitPose(motion, SAMPLE_TIME_MS, motionName);
	}

	private double cartVelocityMmS(double requestedMps) {
		if (Double.isNaN(requestedMps) || Double.isInfinite(requestedMps)
				|| requestedMps <= 0.0) {
			throw new IllegalArgumentException(
					"Capture velocity must be a finite positive value in m/s");
		}

		double requestedMmS = requestedMps * 1000.0;
		double clampedMmS = Math.max(MIN_CART_VEL_MM_S,
				Math.min(MAX_CART_VEL_MM_S, requestedMmS));
		if (clampedMmS != requestedMmS) {
			getLogger().warn("Requested " + requestedMmS
					+ " mm/s; calibration proposal clamps this to "
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
				templateBaseRef);

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
			getLogger().error("Unable to transmit terminal pose: " + e);
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
